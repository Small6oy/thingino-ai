"""PyTorch twin of the persondet JZDL graph, with fake quantisation that follows
libjzdl's integer rules (see docs/jzdl-format.md "Arithmetic").

Real-valued view used here: every activation is u * s, u = unsigned b-bit int
(0 .. 2^b - 1, ReLU built in), s = a learned per-tensor scale. The image is
u = pixel (8 bit), s = 1/255. Convs quantise weights per output channel with a
step chosen so the int16 accumulator bound holds:

    max(|b + umax*sum(w+)|, |b + umax*sum(w-)|) <= ACC_LIMIT, umax = 2^b_in - 1   (int16, see jzdl-format.md)

Heads output float logits: 3 anchors x (5 + num_classes) channels.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

ACC_LIMIT = 32000          # a little under 32767: rounding headroom
# A full conv (JZDL type 46) sums w * (u - 2^(b-1)) (signed input codes) in a wrapping int16 and adds
# its int32 bias afterwards (verified 2026-11-24): bound 2^(b-1) * sum|w| instead, bias outside it.
# Its bias is kept < 2^23 so the float32 QAT conv (sum + bias) stays exact.
BIAS_LIMIT_SIGNED = 1 << 23
WMAX = {'first': 127, 'dw': 127, 'pw': 127, 'head': 127}


def ste_round(x):
    return x + (torch.round(x) - x).detach()


class ActQ(nn.Module):
    """Unsigned b-bit activation with learned scale (LSQ-style)."""

    def __init__(self, bits, init_scale=0.1):
        super().__init__()
        self.bits = bits
        self.log_s = nn.Parameter(torch.tensor(math.log(init_scale)))
        self.enabled = True

    @property
    def scale(self):
        return self.log_s.exp()

    def forward(self, y):
        if not self.enabled:
            return F.relu(y)
        s = self.scale
        q = (2 ** self.bits) - 1
        u = torch.clamp(ste_round(y / s), 0, q)
        return u * s


_FP64 = {}


def f64_device(dev):
    """Where to do the float64 quantisation maths for tensors on dev: dev itself if it has float64,
    else the CPU (Intel Arc A-series has no FP64 at all: any .double() kernel fails). Apple MPS has
    none either; it keeps its float32 path (None) as before."""
    if dev.type == 'mps':
        return None
    if dev.type != 'xpu':
        return dev
    if dev not in _FP64:
        idx = dev.index if dev.index is not None else torch.xpu.current_device()
        _FP64[dev] = dev if getattr(torch.xpu.get_device_properties(idx), 'has_fp64', True) else torch.device('cpu')
    return _FP64[dev]


def weight_step(Wp, B, in_umax, wmax, signed=False):
    """Per-output-channel weight step q (real units per weight LSB).

    Wp: weights already multiplied by the input scale (real per input LSB), [O, ...]
    B:  bias [O] (real). in_umax = 2^b_in - 1.
    q >= max|W'|/wmax  and  sum|W'|/q * in_umax + |B|/q <= ACC_LIMIT
    (signed: type 46, 2^(b-1) * sum|W'|/q <= ACC_LIMIT and |B|/q <= BIAS_LIMIT_SIGNED).
    Uses float64 summation (on the CPU when the GPU has no FP64, e.g. Intel Arc A-series) so reduction-order ULPs
    cannot flip weights sitting on a k + 0.5 rounding boundary between CPU and GPU.
    """
    fd = f64_device(Wp.device)
    use_f64 = fd is not None
    a = Wp.flatten(1).to(fd).double() if use_f64 else Wp.flatten(1)
    Bd = B.to(fd).double() if use_f64 else B
    q1 = a.abs().max(1).values / wmax
    if signed:
        q2 = torch.maximum((in_umax + 1) / 2 * a.abs().sum(1) / ACC_LIMIT, Bd.abs() / BIAS_LIMIT_SIGNED)
    else:
        # inputs are unsigned (0..in_umax): the accumulator spans [B + umax*sum(w-), B + umax*sum(w+)]
        hi = Bd + in_umax * a.clamp(min=0).sum(1)
        lo = Bd + in_umax * a.clamp(max=0).sum(1)
        q2 = torch.maximum(hi.abs(), lo.abs()) / ACC_LIMIT
    q = torch.maximum(torch.maximum(q1, q2), torch.full_like(q1, 1e-12))
    return q.to(Wp.dtype).to(Wp.device) if use_f64 else q


@torch.no_grad()
def mult_shift(M, max_sh=None):
    """requant scale M (0 < M < 1) -> (m, sh), M ~= m / 2^(15 + sh), m in [2^14, 2^15).
    The first conv uses the same m, exported as the Q31 value m << 16.
    max_sh: libjzdl's int16 layers (depthwise, 1x1) take the shift MODULO 16 (verified
    2026-11-25: a shift of 16 acts as 0 and the channel saturates), so they need sh <= 15; a
    smaller M is kept by shrinking m instead (m may reach 0: the channel then outputs its floor)."""
    fd = f64_device(M.device)
    use_f64 = fd is not None
    Md = M.to(fd).double() if use_f64 else M
    Md = Md.clamp(min=1e-30, max=0.99999)
    sh = torch.floor(-torch.log2(Md)).clamp(min=0)
    m = torch.round(Md * torch.pow(2.0, 15 + sh))
    over = m >= 32768
    m = torch.where(over, torch.round(m / 2), m)
    sh = torch.where(over, sh - 1, sh)
    if max_sh is not None:
        cap = sh > max_sh
        m = torch.where(cap, torch.round(m / torch.pow(2.0, sh - max_sh)), m)
        sh = torch.where(cap, torch.full_like(sh, max_sh), sh)
    return (m.to(M.dtype).to(M.device), sh.to(M.dtype).to(M.device)) if use_f64 else (m, sh)


@torch.no_grad()
def acc_bound(w_int, b_int, bits_in, signed=False):
    """per channel worst-case |int16 accumulator| of rounded weights (see weight_step)"""
    umax = 2 ** bits_in - 1
    wf = w_int.flatten(1)
    if signed:
        return torch.maximum((umax + 1) // 2 * wf.abs().sum(1), b_int.abs() * 32767 / BIAS_LIMIT_SIGNED)
    return torch.maximum((b_int + umax * wf.clamp(min=0).sum(1)).abs(), (b_int + umax * wf.clamp(max=0).sum(1)).abs())


@torch.no_grad()
def quant_params(Wp, B, bits_in, wmax, s_out, signed=False, max_sh=None):
    """Per-channel weight step q (int16 bound kept after rounding) and requant (m, sh).
    Shared by training and export so both make identical integer choices. max_sh: see
    mult_shift (MAX_SH[kind])."""
    umax = 2 ** bits_in - 1
    q = weight_step(Wp, B, umax, wmax, signed)
    for _ in range(50):
        ch = acc_bound(torch.round(Wp / q.view(-1, *([1] * (Wp.dim() - 1)))), torch.round(B / q), bits_in, signed)
        over = ch > 32767
        if not over.any():
            break
        q = torch.where(over, q * (ch / 32767 * 1.002), q)
    if s_out is None:
        return q, None, None
    m, sh = mult_shift(q / s_out, max_sh)
    return q, m, sh


MAX_SH = {'first': None, 'dw': 15, 'pw': 15, 'pwpool': 15, 'head': None}   # see mult_shift


if _HAS_TRITON:
    @triton.jit
    def _hw_requant_kernel(t_ptr, m_ptr, sh_ptr, out_ptr, C, HW, n_elements, max_q, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        c = (offs // HW) % C
        t_i = tl.load(t_ptr + offs, mask=mask, other=0.0).to(tl.int64)
        m = tl.load(m_ptr + c, mask=mask, other=0.0).to(tl.int64)
        sh = tl.load(sh_ptr + c, mask=mask, other=0.0).to(tl.int64)
        f = (t_i * m) >> 15
        sh_safe = tl.minimum(tl.maximum(sh, 1), 62)
        one = tl.full([BLOCK], 1, dtype=tl.int64)
        u_shifted = (f + (one << (sh_safe - 1))) >> sh_safe
        u_i = tl.where(sh <= 0, f, tl.where(sh >= 63, 0, u_shifted))
        u_i = tl.minimum(tl.maximum(u_i, 0), max_q)
        tl.store(out_ptr + offs, u_i.to(tl.float32), mask=mask)

    @triton.jit
    def _ste_div_round_scalar_fwd(x_ptr, s_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        sx = tl.load(s_ptr)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.extra.cuda.libdevice.rint(x / sx), mask=mask)

    @triton.jit
    def _ste_div_round_scalar_bwd(go_ptr, s_ptr, gx_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        sx = tl.load(s_ptr)
        go = tl.load(go_ptr + offs, mask=mask, other=0.0)
        tl.store(gx_ptr + offs, go / sx, mask=mask)

    @triton.jit
    def _ste_div_round_chan_fwd(x_ptr, s_ptr, out_ptr, C, HW, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        c = (offs // HW) % C
        sx = tl.load(s_ptr + c, mask=mask, other=1.0)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, tl.extra.cuda.libdevice.rint(x / sx), mask=mask)

    @triton.jit
    def _ste_div_round_chan_bwd(go_ptr, s_ptr, gx_ptr, C, HW, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        c = (offs // HW) % C
        sx = tl.load(s_ptr + c, mask=mask, other=1.0)
        go = tl.load(go_ptr + offs, mask=mask, other=0.0)
        tl.store(gx_ptr + offs, go / sx, mask=mask)

    @triton.jit
    def _fused_post_conv_fwd(
        t_ptr, q_ptr, m_ptr, sh_ptr, s_out_ptr,
        out_ptr, u_hw_ptr, mask_ptr,
        C, HW, n_elements, max_q,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        c = (offs // HW) % C
        t_f = tl.load(t_ptr + offs, mask=mask, other=0.0)
        q = tl.load(q_ptr + c, mask=mask, other=0.0)
        m = tl.load(m_ptr + c, mask=mask, other=0.0).to(tl.int64)
        sh = tl.load(sh_ptr + c, mask=mask, other=0.0).to(tl.int64)
        s_out = tl.load(s_out_ptr)
        t_i = t_f.to(tl.int64)
        f = (t_i * m) >> 15
        sh_safe = tl.minimum(tl.maximum(sh, 1), 62)
        one = tl.full([BLOCK], 1, dtype=tl.int64)
        u_shifted = (f + (one << (sh_safe - 1))) >> sh_safe
        u_i = tl.where(sh <= 0, f, tl.where(sh >= 63, 0, u_shifted))
        u_i = tl.minimum(tl.maximum(u_i, 0), max_q)
        u_f = u_i.to(tl.float32)
        out = u_f * s_out
        v = (t_f * q) / s_out
        # Strict bounds match PyTorch clamp(0, max_q) backward
        in_b = (v > 0.0) & (v < max_q)
        tl.store(out_ptr + offs, out, mask=mask)
        tl.store(u_hw_ptr + offs, u_i.to(tl.uint8), mask=mask)
        tl.store(mask_ptr + offs, in_b.to(tl.uint8), mask=mask)

    @triton.jit
    def _fused_post_conv_bwd(
        go_ptr, t_ptr, q_ptr, u_hw_ptr, mask_ptr, s_out_ptr,
        gt_ptr, gs_part_ptr,
        C, HW, n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        c = (offs // HW) % C
        go = tl.load(go_ptr + offs, mask=mask, other=0.0)
        t_f = tl.load(t_ptr + offs, mask=mask, other=0.0)
        q = tl.load(q_ptr + c, mask=mask, other=0.0)
        u_f = tl.load(u_hw_ptr + offs, mask=mask, other=0).to(tl.float32)
        in_b = tl.load(mask_ptr + offs, mask=mask, other=0) != 0
        s_out = tl.load(s_out_ptr)
        v = (t_f * q) / s_out
        gt = tl.where(in_b, go * q, 0.0)
        gs_el = tl.where(mask, go * tl.where(in_b, u_f - v, u_f), 0.0)
        tl.store(gt_ptr + offs, gt, mask=mask)
        tl.store(gs_part_ptr + pid, tl.sum(gs_el, axis=0))

    class _FusedSteDivRound(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, sx):
            x_c = x.contiguous()
            out = torch.empty_like(x_c)
            n_el = x_c.numel()
            if n_el == 0:
                ctx.save_for_backward(sx)
                ctx.is_scalar = (sx.dim() == 0)
                return out
            grid = ((n_el + 1023) // 1024,)
            if sx.dim() == 0:
                _ste_div_round_scalar_fwd[grid](x_c, sx, out, n_el, BLOCK=1024)
                ctx.is_scalar = True
            else:
                C = x_c.shape[1]
                HW = n_el // (x_c.shape[0] * C)
                _ste_div_round_chan_fwd[grid](x_c, sx.contiguous().view(-1), out, C, HW, n_el, BLOCK=1024)
                ctx.is_scalar = False
                ctx.CHW = (C, HW)
            ctx.save_for_backward(sx)
            return out

        @staticmethod
        def backward(ctx, go):
            (sx,) = ctx.saved_tensors
            go_c = go.contiguous()
            gx = torch.empty_like(go_c)
            n_el = go_c.numel()
            if n_el > 0:
                grid = ((n_el + 1023) // 1024,)
                if ctx.is_scalar:
                    _ste_div_round_scalar_bwd[grid](go_c, sx, gx, n_el, BLOCK=1024)
                else:
                    C, HW = ctx.CHW
                    _ste_div_round_chan_bwd[grid](go_c, sx.contiguous().view(-1), gx, C, HW, n_el, BLOCK=1024)
            return gx, None

    class _FusedPostConv(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t, q, m, sh, s_out, bits):
            t_c = t.contiguous()
            n_el = t_c.numel()
            out = torch.empty_like(t_c)
            if n_el == 0:
                ctx.save_for_backward(t_c, q, torch.empty(0, dtype=torch.uint8, device=t_c.device),
                                      torch.empty(0, dtype=torch.uint8, device=t_c.device), s_out)
                ctx.CHW = (t_c.shape[1], 0)
                return out
            C = t_c.shape[1]
            HW = n_el // (t_c.shape[0] * C)
            u_hw = torch.empty(t_c.shape, dtype=torch.uint8, device=t_c.device)
            in_b = torch.empty(t_c.shape, dtype=torch.uint8, device=t_c.device)
            grid = ((n_el + 1023) // 1024,)
            _fused_post_conv_fwd[grid](
                t_c, q.contiguous(), m.contiguous(), sh.contiguous(), s_out,
                out, u_hw, in_b, C, HW, n_el, (1 << bits) - 1, BLOCK=1024
            )
            ctx.save_for_backward(t_c, q, u_hw, in_b, s_out)
            ctx.CHW = (C, HW)
            return out

        @staticmethod
        def backward(ctx, go):
            t_c, q, u_hw, in_b, s_out = ctx.saved_tensors
            go_c = go.contiguous()
            n_el = go_c.numel()
            gt = torch.empty_like(go_c)
            if n_el > 0:
                n_blocks = (n_el + 1023) // 1024
                gs_part = torch.empty((n_blocks,), dtype=torch.float32, device=go_c.device)
                C, HW = ctx.CHW
                _fused_post_conv_bwd[(n_blocks,)](
                    go_c, t_c, q.contiguous(), u_hw, in_b, s_out,
                    gt, gs_part, C, HW, n_el, BLOCK=1024
                )
                gs = gs_part.sum()
            else:
                gs = torch.zeros_like(s_out)
            return gt, None, None, None, gs, None


def hw_requant(t, m, sh, bits, wide=False):
    """Exact libjzdl requant in float32: clamp(rsr(floor(t*m / 2^15), sh), 0, 2^bits - 1).
    t: integer-valued tensor |t| <= 32767 (N,C,H,W); m, sh: per-channel (C,). m is split
    so every intermediate stays below 2^24 (exact in float32). wide: |t| < 2^24 (type 46:
    int16 sum + int32 bias), done in int64."""
    if _HAS_TRITON and t.is_cuda and t.dtype == torch.float32 and t.dim() >= 2:
        t_c = t.contiguous()
        n_el = t_c.numel()
        out = torch.empty_like(t_c)
        if n_el > 0:
            C = t_c.shape[1]
            HW = n_el // (t_c.shape[0] * C)
            _hw_requant_kernel[((n_el + 1023) // 1024,)](
                t_c, m.contiguous(), sh.contiguous(), out, C, HW, n_el, (1 << bits) - 1, BLOCK=1024
            )
        return out
    m = m.view(1, -1, 1, 1); sh = sh.view(1, -1, 1, 1)
    if wide:
        f = torch.div(t.round().long() * m.long(), 32768, rounding_mode='floor')
        si = sh.long()
        half = torch.where(si > 0, torch.bitwise_left_shift(torch.ones_like(si), (si - 1).clamp(min=0)), torch.zeros_like(si))
        u = torch.div(f + half, torch.bitwise_left_shift(torch.ones_like(si), si), rounding_mode='floor')
        return u.clamp(0, 2 ** bits - 1).to(t.dtype)
    m_hi = torch.floor(m / 256); m_lo = m - 256 * m_hi
    A = t * m_hi; Bv = t * m_lo
    a1 = torch.floor(A / 128); a0 = A - 128 * a1
    f = a1 + torch.floor((a0 * 256 + Bv) / 32768)
    p = torch.pow(2.0, sh)
    u = torch.where(sh > 0, torch.floor((f + p / 2) / p), f)
    return u.clamp(0, 2 ** bits - 1)


class QConv(nn.Module):
    """kind: first (full k x k conv, JZDL type 46: the 3x3 s2 on the image, or anywhere in a graph),
    dw (3x3 depthwise), pw (1x1), pwpool (1x1 + 2x2 maxpool), head (1x1, float out).
    k / conv_dil: kernel size and dilation of a full conv (dilation is exported as a zero-stuffed
    (k-1)*d+1 kernel: libjzdl has no dilation). acc_signed: the type 46 accumulator rule (signed input
    codes, int32 bias outside the int16 sum; see weight_step); same_pad: TF 'SAME' padding (type 46)."""

    def __init__(self, kind, cin, cout, bits_in, bits_out=None, stride=1, k=3):
        super().__init__()
        self.kind, self.cin, self.cout, self.bits_in, self.bits_out, self.stride = kind, cin, cout, bits_in, bits_out, stride
        if kind == 'dw':
            self.weight = nn.Parameter(torch.randn(cout, 1, 3, 3) * math.sqrt(2 / 9))
        elif kind == 'first':
            self.weight = nn.Parameter(torch.randn(cout, cin, k, k) * math.sqrt(2 / (cin * k * k)))
        else:
            self.weight = nn.Parameter(torch.randn(cout, cin, 1, 1) * math.sqrt(2 / cin))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.bn = nn.BatchNorm2d(cout) if kind != 'head' else None
        self.act = ActQ(bits_out) if kind != 'head' else None
        self.quant_w = True
        self.bn_folded = False
        # geometry (persondet defaults; graphnet.py sets other strides/pads from the JZDL params)
        self.conv_stride = 2 if kind == 'first' else 1
        self.conv_pad = (0, 1, 0, 1) if kind == 'first' else (1, 1, 1, 1) if kind == 'dw' else (0, 0, 0, 0)
        self.conv_dil = 1
        self.acc_signed = False
        self.same_pad = False

    def folded(self):
        """Weights/bias with BatchNorm folded in (export and QAT both use this)."""
        W, B = self.weight, self.bias
        if self.bn is not None and not self.bn_folded:
            bn = self.bn
            g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            W = W * g.view(-1, 1, 1, 1)
            B = (B - bn.running_mean) * g + bn.bias
        return W, B

    def in_scale(self, s_in):
        """broadcastable input scale (per-tensor scalar or per-channel vector for concat inputs)"""
        if s_in.dim() == 0:
            return s_in
        return s_in.view(1, -1, 1, 1) if self.kind != 'dw' else s_in.view(-1, 1, 1, 1)

    def forward(self, x, s_in):
        """x: real activations (u * s_in). s_in: input scale tensor."""
        use_bn_stats = self.training and self.bn is not None and not self.bn_folded
        if use_bn_stats:
            # plain conv + BN in training (fold at the end of QAT warmup / export)
            y = self._conv(x, self.weight, self.bias)
            y = self.bn(y)
        else:
            W, B = self.folded()
            if not self.quant_w:
                y = self._conv(x, W, B)
            else:
                return self._quant_forward(x, s_in, W, B)
        if self.kind == 'head':
            return y
        y = self.act(y)
        if self.kind == 'pwpool':
            y = F.max_pool2d(y, 2, 2)
        return y

    def _quant_forward(self, x, s_in, W, B):
        """Integer-domain forward that matches libjzdl exactly (STE gradients)."""
        si = self.in_scale(s_in.detach())              # real value per input LSB, weight-shaped
        sx = s_in.detach() if s_in.dim() == 0 else s_in.detach().view(1, -1, 1, 1)
        use_fused = _HAS_TRITON and x.is_cuda and x.dtype == torch.float32
        u_in = _FusedSteDivRound.apply(x, sx) if use_fused else ste_round(x / sx)  # unsigned input codes
        Wp = W * si
        wmax = WMAX[self.kind if self.kind != 'pwpool' else 'pw']
        s_out = self.act.scale if self.act is not None else None
        q, m, sh = quant_params(Wp.detach(), B.detach(), self.bits_in, wmax,
                                None if s_out is None else s_out.detach(), self.acc_signed, MAX_SH[self.kind])
        wide = self.acc_signed                         # acc + int32 bias can pass 32767
        w_int = ste_round(Wp / q.view(-1, 1, 1, 1))
        b_int = ste_round(B / q)
        t = self._conv(u_in, w_int, b_int)             # = acc_hw + bias_hw, exact integers
        if self.kind == 'head':
            return t * q.view(1, -1, 1, 1)             # real pre-activation
        if use_fused and torch.is_grad_enabled() and self.bits_out <= 8:
            out = _FusedPostConv.apply(t, q, m, sh, s_out, self.bits_out)
        elif use_fused and not torch.is_grad_enabled():
            out = hw_requant(t, m, sh, self.bits_out, wide) * s_out
        else:
            y = t * q.view(1, -1, 1, 1)                # real pre-activation
            v = y / s_out
            u_hw = hw_requant(t.detach(), m, sh, self.bits_out, wide)
            vc = v.clamp(0, 2 ** self.bits_out - 1)
            u = u_hw + (vc - vc.detach())              # forward = hardware, backward = LSQ/STE
            out = u * s_out
        if self.kind == 'pwpool':
            out = F.max_pool2d(out, 2, 2)
        return out

    def _conv(self, x, W, B):
        pad = self.conv_pad
        if self.same_pad:                               # libjzdl type 46: TF 'SAME', whatever the file says
            ke = (W.shape[-1] - 1) * self.conv_dil + 1
            (pt, pb), (pl, pr) = (same_pad(n, ke, self.conv_stride) for n in x.shape[2:])
            pad = (pl, pr, pt, pb)
        if any(pad):                                    # (left, right, top, bottom), zeros
            x = F.pad(x, pad)
        return F.conv2d(x, W, B, stride=self.conv_stride, dilation=self.conv_dil, groups=self.cout if self.kind == 'dw' else 1)


def same_pad(n, k, s):
    """TF 'SAME' padding (before, after) of a length-n axis: output ceil(n / s), odd extra after"""
    tot = max((-(-n // s) - 1) * s + k - n, 0)
    return tot // 2, tot - tot // 2


def zero_insert_up(x):
    n, c, h, w = x.shape
    mf = torch.channels_last if x.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
    z = torch.empty((n, c, h * 2, w * 2), dtype=x.dtype, device=x.device, memory_format=mf).zero_()
    z[:, :, ::2, ::2] = x
    return z


class Detector(nn.Module):
    """persondet topology. widths can be scaled; num_classes sets the heads (3 x (5 + N))."""

    def __init__(self, num_classes=3, anchors=3, width=1.0):
        super().__init__()
        c = lambda n: max(16, int(round(n * width / 16)) * 16)
        self.num_classes, self.anchors = num_classes, anchors
        no = anchors * (5 + num_classes)
        self.no = no
        C = dict(c16=c(16), c32=c(32), c64=c(64), c128=c(128), c256=c(256), c512=c(512))
        self.C = C
        L = nn.ModuleDict()
        L['l1'] = QConv('first', 3, C['c16'], 8, 5)
        L['l2'] = QConv('dw', C['c16'], C['c16'], 5, 5)
        L['l3'] = QConv('pwpool', C['c16'], C['c32'], 5, 5)
        L['l4'] = QConv('dw', C['c32'], C['c32'], 5, 5)
        L['l5'] = QConv('pwpool', C['c32'], C['c64'], 5, 5)
        L['l6'] = QConv('dw', C['c64'], C['c64'], 5, 5)
        L['l7'] = QConv('pwpool', C['c64'], C['c128'], 5, 4)
        L['l8'] = QConv('dw', C['c128'], C['c128'], 4, 4)
        L['l9'] = QConv('pw', C['c128'], C['c256'], 4, 4)
        L['l12'] = QConv('dw', C['c256'], C['c256'], 4, 4)
        L['l13'] = QConv('pw', C['c256'], C['c512'], 4, 4)
        L['l14'] = QConv('dw', C['c512'], C['c512'], 4, 4)
        L['l15'] = QConv('pw', C['c512'], C['c512'], 4, 4)
        L['l16'] = QConv('pw', C['c512'], C['c256'], 4, 4)
        L['l18'] = QConv('dw', C['c256'], C['c256'], 4, 4)
        L['l19'] = QConv('pw', C['c256'], C['c512'], 4, 4)
        L['l20'] = QConv('head', C['c512'], no, 4)
        L['l21'] = QConv('pw', C['c256'], C['c128'], 4, 4)
        L['l24'] = QConv('dw', C['c128'] + C['c256'], C['c128'] + C['c256'], 4, 4)
        L['l25'] = QConv('pw', C['c128'] + C['c256'], C['c256'], 4, 4)
        L['l26'] = QConv('head', C['c256'], no, 4)
        self.L = L
        self.in_scale = 1.0 / 255.0
        self.register_buffer('_in_scale_t', torch.tensor(self.in_scale, dtype=torch.float32), persistent=False)

    def forward(self, img):
        """img: float tensor N x 3 x H x W holding pixel values 0..255. Returns (head32, head16) logits NCHW."""
        L = self.L
        s = self._in_scale_t if self._in_scale_t.device == img.device else self._in_scale_t.to(img.device)
        x = img * self.in_scale
        sc = lambda name: L[name].act.scale
        x = L['l1'](x, s); x = L['l2'](x, sc('l1')); x = L['l3'](x, sc('l2')); x = L['l4'](x, sc('l3'))
        x = L['l5'](x, sc('l4')); x = L['l6'](x, sc('l5')); x = L['l7'](x, sc('l6')); x = L['l8'](x, sc('l7'))
        b9 = L['l9'](x, sc('l8'))
        x = F.max_pool2d(b9, 2, 2)
        x = L['l12'](x, sc('l9')); x = L['l13'](x, sc('l12')); x = L['l14'](x, sc('l13'))
        x = L['l15'](x, sc('l14')); b17 = L['l16'](x, sc('l15'))
        y = L['l18'](b17, sc('l16')); y = L['l19'](y, sc('l18')); h32 = L['l20'](y, sc('l19'))
        z = L['l21'](b17, sc('l16')); z = zero_insert_up(z)
        z = torch.cat([z, b9], 1)
        s_cat = torch.cat([sc('l21').expand(self.C['c128']), sc('l9').expand(self.C['c256'])])
        z = L['l24'](z, s_cat); z = L['l25'](z, sc('l24')); h16 = L['l26'](z, sc('l25'))
        return h32, h16

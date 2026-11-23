"""Designed experiments against libjzdl in the offline runtime (QEMU on the build host, $JZDL_BUILD_HOST).

Build small edited models (one-hot weights, chosen bias/mult/shift), run a batch
under ~/jzdl-qemu/run/batch.sh on the build host, get the probed blobs back.
This is how the arithmetic in ../../jzdl-format.md was decoded. Work files go to
$JZDL_WORK (default: /tmp/jzdl-exp). Usage: see jzdl-format.md "Arithmetic".
"""
import numpy as np, struct, sys, subprocess, os, shutil, copy
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import jzdl, probe
S=os.environ.get('JZDL_WORK','/tmp/jzdl-exp')
HOST=os.environ.get('JZDL_BUILD_HOST')
if not HOST:
    raise SystemExit('set JZDL_BUILD_HOST to the build host (ssh name or address) that has ~/jzdl-qemu')
BASE=jzdl.Model.parse(open(HERE+'/../aitest/out/v3/magik_model_persondet.bin','rb').read())
def l1_model(H=16,W=16,weights=None,bias=None,mult=None,shift=None,bits=None,params=None):
    m=copy.deepcopy(BASE); m.in_h,m.in_w=H,W
    L=m.layers[1]
    if weights is not None: L.blobs[0]=np.asarray(weights,np.int8).tobytes()
    if bias is not None: L.blobs[1]=np.asarray(bias,np.int32).tobytes()
    if mult is not None: L.blobs[2]=np.asarray(mult,np.int32).tobytes()
    if shift is not None: L.blobs[3]=np.asarray(shift,np.int32).tobytes()
    if params: 
        for k,v in params.items(): L.params[k]=v
    return probe.make_probe(m,1).to_bytes()
def run(batch, tag):
    """batch: name -> (model_bytes, rgb uint8 HxWx3). returns name -> HWC int array"""
    d=S+'/batch/'+tag; shutil.rmtree(d,ignore_errors=True); os.makedirs(d)
    for n,(mb,img) in batch.items():
        open(d+'/%s.bin'%n,'wb').write(mb); open(d+'/%s.rgb'%n,'wb').write(np.asarray(img,np.uint8).tobytes())
    r='jzdl-qemu/run/batch/'+tag
    subprocess.run(['ssh','-o','BatchMode=yes',HOST,'rm -rf ~/%s; mkdir -p ~/%s'%(r,r)],check=True)
    subprocess.run('scp -q %s/* %s:%s/'%(d,HOST,r),shell=True,check=True)
    p=subprocess.run(['ssh','-o','BatchMode=yes',HOST,'~/jzdl-qemu/run/batch.sh ~/%s'%r],capture_output=True,text=True); print(p.stdout,p.stderr,end='')
    subprocess.run('scp -q "%s:%s/*.f32" "%s:%s/*.log" %s/'%(HOST,r,HOST,r,d),shell=True)
    res={}
    for n in batch:
        f=d+'/%s.out.-999.f32'%n
        if not os.path.exists(f): print('no output',n,open(d+'/%s.log'%n).read()[-300:]); continue
        w,h,c=np.fromfile(f,np.int32,count=3); a=np.fromfile(f,np.float32,offset=12)
        res[n]=a.reshape(h,w,c)
    return res

def l2_model(H=32,W=32,weights=None,bias=None,mult=None,shift=None,a4=None,params=None):
    """Input -> persondet layer1 (unchanged) -> layer2 (t73 dw, edited) -> probe on blob 2"""
    m=copy.deepcopy(BASE); m.in_h,m.in_w=H,W
    L=m.layers[2]
    for k,v in ((0,weights),(1,bias),(2,mult),(3,shift),(4,a4)):
        if v is not None: L.blobs[k]=np.asarray(v,np.int8 if k==0 else np.int16).tobytes()
    if params:
        for k,v in params.items(): L.params[k]=v
    return probe.make_probe(m,2).to_bytes()

def ref_l1(rgb):
    """bit-exact persondet layer 1 (verified 2026-11-23): rgb HxWx3 uint8 -> int (u-16) H/2 x W/2 x 16"""
    L=BASE.layers[1]; Wt=np.frombuffer(L.blobs[0],np.int8).astype(np.int64)
    bias=np.frombuffer(L.blobs[1],np.int32).astype(np.int64); mult=np.frombuffer(L.blobs[2],np.int32).astype(np.int64)
    sh=np.frombuffer(L.blobs[3],np.int32).astype(np.int64)
    H,Wd,_=rgb.shape; x=rgb.astype(np.int64)-128
    xp=np.full((H+1,Wd+1,3),-128,np.int64); xp[:H,:Wd]=x
    acc=np.zeros((H//2,Wd//2,16),np.int64)
    for ky in range(3):
        for kx in range(3):
            wk=np.array([[Wt[((ky*10+kx*3+ci)//2)*32+o*2+(ky*10+kx*3+ci)%2] for ci in range(3)] for o in range(16)])
            acc+=np.einsum('hwc,oc->hwo',xp[ky:ky+H:2,kx:kx+Wd:2],wk)
    v=((acc+bias)*mult)>>31
    v=np.where(sh>0,(v+(1<<(np.maximum(sh,1)-1)))>>sh,v)
    return np.clip(v,0,31)-16

def edit_model(li,H=32,W=32,blobs=None,params=None,base=None):
    """copy of BASE (input HxW) with layer li's blobs/params replaced; probe on its first top"""
    m=copy.deepcopy(base or BASE); m.in_h,m.in_w=H,W
    L=m.layers[li]
    dt={0:np.int8}
    if blobs:
        for k,v in blobs.items():
            if v is not None: L.blobs[k]=np.asarray(v,dt.get(k, np.int16 if L.type in (53,73,74) else np.int32)).tobytes()
    if params:
        for k,v in params.items(): L.params[k]=v
    return probe.make_probe(m,L.tops[0]).to_bytes()
def in_probe(li,H=32,W=32,base=None):
    m=copy.deepcopy(base or BASE); m.in_h,m.in_w=H,W
    return probe.make_probe(m,m.layers[li].bottoms[0]).to_bytes()

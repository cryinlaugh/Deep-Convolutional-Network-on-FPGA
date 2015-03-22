import os
import shutil
import random
import cPickle
import time
import math

PROJ_NAME = None

def del_slic():
    fns = ['%s.py' %(PROJ_NAME),'_%s.so' %(PROJ_NAME)]
    dns = ['simutils']
    for fn in fns:
        os.remove(fn)
    for dn in dns:
        shutil.rmtree(dn)

def gen_slic(ver='Simulation'):
    os.system('sliccompile -t python -m ../RunRules/%s/maxfiles/%s.max' %(ver,PROJ_NAME))

def fix(x,s):
    l0 = len(x)
    l1 = (l0+s-1)/s*s
    if l0!=l1:
        print 'fix length from %d to %d' %(l0,l1)
        return x+[0.0]*(l1-l0)
    return x

def check(name,eps,res,f):
    cnt = len([1 for i in xrange(len(res)) if abs(res[i]-f[i])>eps])
    print 'len(res_%s) = %d ; %d = len(std_%s)' %(name,len(res),len(f),name)
    print '# diff > %.15f : %d / %d' %(eps,cnt,len(res))
    print 'max diff = %.15f' %(max([abs(res[i]-f[i]) for i in xrange(len(res))]))
    return (len(res)==len(f)) and (cnt==0)

def gen_conv_data(filename='data.bin',ni=20,no=50,row=12,col=12,k=5,batch_size=384):
    t0 = time.time()
    random.seed()
    para = [ni,no,row,col,k,batch_size]
    b = [random.random() for i2 in xrange(no)]
    w = [[[[random.random() for i6 in xrange(k)] for i5 in xrange(k)] for i2 in xrange(no)] for i1 in xrange(ni)]
    x = [[[[random.random() for i7 in xrange(batch_size)] for i4 in xrange(col)] for i3 in xrange(row)] for i1 in xrange(ni)]
    z = [[[[sum([w[i1][i2][i5][i6]*x[i1][i3+i5][i4+i6][i7] for i1 in xrange(ni) for i5 in xrange(k) for i6 in xrange(k)])+b[i2] for i7 in xrange(batch_size)] for i4 in xrange(col-k+1)] for i3 in xrange(row-k+1)] for i2 in xrange(no)]
    z_grad = [[[[random.random() for i7 in xrange(batch_size)] for i4 in xrange(col-k+1)] for i3 in xrange(row-k+1)] for i2 in xrange(no)]
    x_grad = [[[[
        sum([
            w[i1][i2][i5][i6]*z_grad[i2][i3-i5][i4-i6][i7]
            for i2 in xrange(no) for i5 in xrange(k) for i6 in xrange(k) if (i3-i5>=0)and(i3-i5<=row-k)and(i4-i6>=0)and(i4-i6<=col-k)
        ])
        for i7 in xrange(batch_size)] for i4 in xrange(col)] for i3 in xrange(row)] for i1 in xrange(ni)]
    w_grad = [[[[[
        sum([
            x[i1][i3+i5][i4+i6][i7]*z_grad[i2][i3][i4][i7]
            for i3 in xrange(row-k+1) for i4 in xrange(col-k+1)
        ])
        for i7 in xrange(batch_size)] for i6 in xrange(k)] for i5 in xrange(k)] for i2 in xrange(no)] for i1 in xrange(ni)]
    fb = [v for v in b for i3 in xrange(row-k+1) for i4 in xrange(col-k+1) for i7 in xrange(batch_size)]
    fw = [w[i1][i2][i5][i6] for i1 in xrange(ni) for i2 in xrange(no) for i5 in xrange(k) for i6 in xrange(k)]
    fx = [x[i1][i3][i4][i7] for i1 in xrange(ni) for i3 in xrange(row) for i4 in xrange(col) for i7 in xrange(batch_size)]
    fz = [z[i2][i3][i4][i7] for i2 in xrange(no) for i3 in xrange(row-k+1) for i4 in xrange(col-k+1) for i7 in xrange(batch_size)]
    fz_grad = [z_grad[i2][i3][i4][i7] for i2 in xrange(no) for i3 in xrange(row-k+1) for i4 in xrange(col-k+1) for i7 in xrange(batch_size)]
    fx_grad = [x_grad[i1][i3][i4][i7] for i1 in xrange(ni) for i3 in xrange(row) for i4 in xrange(col) for i7 in xrange(batch_size)]
    fw_grad = [w_grad[i1][i2][i5][i6][i7] for i1 in xrange(ni) for i2 in xrange(no) for i5 in xrange(k) for i6 in xrange(k) for i7 in xrange(batch_size)]
    with open(filename,'w') as ouf:
        cPickle.dump(para,ouf)
        cPickle.dump(fb,ouf)
        cPickle.dump(fw,ouf)
        cPickle.dump(fx,ouf)
        cPickle.dump(fz,ouf)
        cPickle.dump(fz_grad,ouf)
        cPickle.dump(fx_grad,ouf)
        cPickle.dump(fw_grad,ouf)

        cPickle.dump(b,ouf)
        cPickle.dump(w,ouf)
        cPickle.dump(x,ouf)
        cPickle.dump(z,ouf)
        cPickle.dump(z_grad,ouf)
        cPickle.dump(x_grad,ouf)
        cPickle.dump(w_grad,ouf)
    print '[INFO] time used = %f' %(time.time()-t0)
    return dict(para=para,fb=fb,fw=fw,fx=fx,fz=fz,fz_grad=fz_grad,fx_grad=fx_grad,fw_grad=fw_grad)

def gen_max_pool_data(filename='data.bin',no=50,row=8,col=8,k=2,batch_size=384):
    t0 = time.time()
    random.seed()
    para = [no,row,col,k,batch_size]
    z2 = [[[[float(random.randint(0,100)) for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    z = [[[[max([z2[i1][i2*k+i5][i3*k+i6][i4] for i5 in xrange(k) for i6 in xrange(k)]) for i4 in xrange(batch_size)] for i3 in xrange(col/k)] for i2 in xrange(row/k)] for i1 in xrange(no)]
    sels = [[[[1 if z2[i1][i2][i3][i4]==z[i1][i2/k][i3/k][i4] else 0 for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    sel = [[[[
        [sels[i1][i2+i5][i3+i6][i4] for i5 in xrange(k) for i6 in xrange(k)]
        for i4 in xrange(batch_size)] for i3 in xrange(0,col,k)] for i2 in xrange(0,row,k)] for i1 in xrange(no)]
    z_grad = [[[[random.random() for i4 in xrange(batch_size)] for i3 in xrange(col/k)] for i2 in xrange(row/k)] for i1 in xrange(no)]
    z2_grad = [[[[z_grad[i1][i2/k][i3/k][i4] if sels[i1][i2][i3][i4]==1 else 0. for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    fz2 = [z2[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row) for i3 in xrange(col) for i4 in xrange(batch_size)]
    fz = [z[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row/k) for i3 in xrange(col/k) for i4 in xrange(batch_size)]
    fsel = [sels[i1][i2+i5][i3+i6][i4] for i1 in xrange(no) for i2 in xrange(0,row,k) for i3 in xrange(0,col,k) for i4 in xrange(batch_size) for i5 in xrange(k) for i6 in xrange(k)]
    fselw = [sum([fsel[i+b]<<b for b in xrange(8)]) for i in xrange(0,len(fsel),8)]
    fz_grad = [z_grad[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row/k) for i3 in xrange(col/k) for i4 in xrange(batch_size)]
    fz2_grad = [z2_grad[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row) for i3 in xrange(col) for i4 in xrange(batch_size)]
    with open(filename,'w') as ouf:
        cPickle.dump(para,ouf)
        cPickle.dump(fz2,ouf)
        cPickle.dump(fz,ouf)
        cPickle.dump(fselw,ouf)
        cPickle.dump(fz_grad,ouf)
        cPickle.dump(fz2_grad,ouf)

        cPickle.dump(z2,ouf)
        cPickle.dump(z,ouf)
        cPickle.dump(sel,ouf)
        cPickle.dump(z_grad,ouf)
        cPickle.dump(z2_grad,ouf)
    print '[INFO] time used = %f' %(time.time()-t0)
    return dict(para=para,fz2=fz2,fz=fz,fselw=fselw,fz_grad=fz_grad,fz2_grad=fz2_grad)

def gen_tanh_data(filename='data.bin',no=50,row=4,col=4,batch_size=384):
    t0 = time.time()
    random.seed()
    para = [no,row,col,batch_size]
    fz = [random.random() for i1 in xrange(no) for i2 in xrange(row) for i3 in xrange(col) for i4 in xrange(batch_size)]
    cosm2 = [math.exp(z)+math.exp(-z) for z in fz]
    fa = [(math.exp(z)-math.exp(-z))/cosm2[i] for i,z in enumerate(fz)]
    fa_grad = [random.random() for i1 in xrange(no) for i2 in xrange(row) for i3 in xrange(col) for i4 in xrange(batch_size)]
    fz_grad = [4./x/x*fa_grad[i] for i,x in enumerate(cosm2)]
    with open(filename,'w') as ouf:
        cPickle.dump(para,ouf)
        cPickle.dump(fz,ouf)
        cPickle.dump(fa,ouf)
        cPickle.dump(fa_grad,ouf)
        cPickle.dump(fz_grad,ouf)
    print '[INFO] time used = %f' %(time.time()-t0)
    return dict(para=para,fz=fz,fa=fa,fa_grad=fa_grad,fz_grad=fz_grad)

def gen_softmax_data(filename='data.bin',ni=500,no=10,batch_size=384):
    t0 = time.time()
    random.seed()
    para = [ni,no,batch_size]
    b = [random.random() for i2 in xrange(no)]
    x = [[random.random() for i3 in xrange(batch_size)] for i1 in xrange(ni)]
    w = [[random.random() for i2 in xrange(no)] for i1 in xrange(ni)]
    z = [[sum([x[i1][i3]*w[i1][i2] for i1 in xrange(ni)])+b[i2] for i3 in xrange(batch_size)] for i2 in xrange(no)]
    maxz = [max([z[i2][i3] for i2 in xrange(no)]) for i3 in xrange(batch_size)]
    expz = [[math.exp(z[i2][i3]-maxz[i3]) for i3 in xrange(batch_size)] for i2 in xrange(no)]
    sumexpz = [sum([expz[i2][i3] for i2 in xrange(no)]) for i3 in xrange(batch_size)]
    sm = [[expz[i2][i3]/sumexpz[i3] for i3 in xrange(batch_size)] for i2 in xrange(no)]
    pred = [max([(sm[i2][i3],i2) for i2 in xrange(no)],key=lambda x:x[0])[1] for i3 in xrange(batch_size)]
    std = [random.randint(0,no-1) for i3 in xrange(batch_size)]
    delta = [[1. if std[i3]==i2 else 0. for i3 in xrange(batch_size)] for i2 in xrange(no)]
    b_grad = [[sm[i2][i3]-delta[i2][i3] for i3 in xrange(batch_size)] for i2 in xrange(no)]
    w_grad = [[[x[i1][i3]*(sm[i2][i3]-delta[i2][i3]) for i3 in xrange(batch_size)] for i2 in xrange(no)] for i1 in xrange(ni)]
    z_grad = [[(sm[i2][i3]-delta[i2][i3])/batch_size for i3 in xrange(batch_size)] for i2 in xrange(no)]
    x_grad = [[sum([z_grad[i2][i3]*x[i1][i3] for i2 in xrange(no)]) for i3 in xrange(batch_size)] for i1 in xrange(ni)]
    fb = b
    fx = [x[i1][i3] for i1 in xrange(ni) for i3 in xrange(batch_size)]
    fw = [w[i1][i2] for i1 in xrange(ni) for i2 in xrange(no)]
    fz = [z[i2][i3] for i2 in xrange(no) for i3 in xrange(batch_size)]
    fsm = [sm[i2][i3] for i2 in xrange(no) for i3 in xrange(batch_size)]
    fpred = pred
    fstd = std
    fb_grad = [b_grad[i2][i3] for i2 in xrange(no) for i3 in xrange(batch_size)]
    fw_grad = [w_grad[i1][i2][i3] for i1 in xrange(ni) for i2 in xrange(no) for i3 in xrange(batch_size)]
    fx_grad = [x_grad[i1][i3] for i1 in xrange(ni) for i3 in xrange(batch_size)]
    with open(filename,'w') as ouf:
        cPickle.dump(para,ouf)
        cPickle.dump(fb,ouf)
        cPickle.dump(fx,ouf)
        cPickle.dump(fw,ouf)
        cPickle.dump(fz,ouf)
        cPickle.dump(fsm,ouf)
        cPickle.dump(fpred,ouf)
        cPickle.dump(fstd,ouf)
        cPickle.dump(fb_grad,ouf)
        cPickle.dump(fw_grad,ouf)
        cPickle.dump(fx_grad,ouf)

        cPickle.dump(b,ouf)
        cPickle.dump(x,ouf)
        cPickle.dump(w,ouf)
        cPickle.dump(z,ouf)
        cPickle.dump(sm,ouf)
        cPickle.dump(pred,ouf)
        cPickle.dump(std,ouf)
        cPickle.dump(b_grad,ouf)
        cPickle.dump(w_grad,ouf)
        cPickle.dump(x_grad,ouf)
    print '[INFO] time used = %f' %(time.time()-t0)
    return dict(para=para,fb=fb,fx=fx,fw=fw,fz=fz,fsm=fsm,fpred=fpred,fstd=fstd,fb_grad=fb_grad,fw_grad=fw_grad,fx_grad=fx_grad)


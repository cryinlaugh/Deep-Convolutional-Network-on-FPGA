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
    print 'value range: res = (%.15f,%.15f) ; (%.15f,%.15f) = std' %(min(res),max(res),min(f),max(f))
    return (len(res)==len(f)) and (cnt==0)

def gen_conv_data(filename='data.bin',ni=20,no=50,row=12,col=12,k=5,batch_size=384):
    t0 = time.time()
    random.seed()
    para = [ni,no,row,col,k,batch_size]
    b = [random.random()*2.-1. for i2 in xrange(no)]
    w = [[[[random.random()*2.-1. for i6 in xrange(k)] for i5 in xrange(k)] for i2 in xrange(no)] for i1 in xrange(ni)]
    x = [[[[random.random()*2.-1. for i7 in xrange(batch_size)] for i4 in xrange(col)] for i3 in xrange(row)] for i1 in xrange(ni)]
    z = [[[[sum([w[i1][i2][i5][i6]*x[i1][i3+i5][i4+i6][i7] for i1 in xrange(ni) for i5 in xrange(k) for i6 in xrange(k)])+b[i2] for i7 in xrange(batch_size)] for i4 in xrange(col-k+1)] for i3 in xrange(row-k+1)] for i2 in xrange(no)]
    z_grad = [[[[random.random()*2.-1. for i7 in xrange(batch_size)] for i4 in xrange(col-k+1)] for i3 in xrange(row-k+1)] for i2 in xrange(no)]
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
    z2 = [[[[float(random.randint(0,200)-100) for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    z = [[[[max([z2[i1][i2*k+i5][i3*k+i6][i4] for i5 in xrange(k) for i6 in xrange(k)]) for i4 in xrange(batch_size)] for i3 in xrange(col/k)] for i2 in xrange(row/k)] for i1 in xrange(no)]
    sels = [[[[1 if z2[i1][i2][i3][i4]==z[i1][i2/k][i3/k][i4] else 0 for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    sel = [[[[
        [sels[i1][i2+i5][i3+i6][i4] for i5 in xrange(k) for i6 in xrange(k)]
        for i4 in xrange(batch_size)] for i3 in xrange(0,col,k)] for i2 in xrange(0,row,k)] for i1 in xrange(no)]
    a = [[[[(math.exp(t)-math.exp(-t))/(math.exp(t)+math.exp(-t)) for t in z[i1][i2][i3]] for i3 in xrange(col/k)] for i2 in xrange(row/k)] for i1 in xrange(no)]
    a_grad = [[[[random.random()*2.-1. for i4 in xrange(batch_size)] for i3 in xrange(col/k)] for i2 in xrange(row/k)] for i1 in xrange(no)]
    z_grad = [[[[4.*a_grad[i1][i2][i3][i4]/(math.exp(t)+math.exp(-t))/(math.exp(t)+math.exp(-t)) for i4,t in enumerate(z[i1][i2][i3])] for i3 in xrange(col/k)] for i2 in xrange(row/k)] for i1 in xrange(no)]
    z2_grad = [[[[z_grad[i1][i2/k][i3/k][i4] if sels[i1][i2][i3][i4]==1 else 0. for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    
    fz2 = [z2[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row) for i3 in xrange(col) for i4 in xrange(batch_size)]
    fz = [z[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row/k) for i3 in xrange(col/k) for i4 in xrange(batch_size)]
    fsel = [sels[i1][i2+i5][i3+i6][i4] for i1 in xrange(no) for i2 in xrange(0,row,k) for i3 in xrange(0,col,k) for i4 in xrange(batch_size) for i5 in xrange(k) for i6 in xrange(k)]
    fselw = [sum([fsel[i+b]<<b for b in xrange(8)]) for i in xrange(0,len(fsel),8)]
    fa = [a[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row/k) for i3 in xrange(col/k) for i4 in xrange(batch_size)]
    fa_grad = [a_grad[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row/k) for i3 in xrange(col/k) for i4 in xrange(batch_size)]
    fz_grad = [z_grad[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row/k) for i3 in xrange(col/k) for i4 in xrange(batch_size)]
    fz2_grad = [z2_grad[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row) for i3 in xrange(col) for i4 in xrange(batch_size)]
    with open(filename,'w') as ouf:
        cPickle.dump(para,ouf)
        cPickle.dump(fz2,ouf)
        cPickle.dump(fselw,ouf)
        cPickle.dump(fz,ouf)
        cPickle.dump(fa,ouf)
        cPickle.dump(fa_grad,ouf)
        cPickle.dump(fz_grad,ouf)
        cPickle.dump(fz2_grad,ouf)

        cPickle.dump(z2,ouf)
        cPickle.dump(sel,ouf)
        cPickle.dump(z,ouf)
        cPickle.dump(a,ouf)
        cPickle.dump(a_grad,ouf)
        cPickle.dump(z_grad,ouf)
        cPickle.dump(z2_grad,ouf)
    print '[INFO] time used = %f' %(time.time()-t0)
    return dict(para=para,fz2=fz2,fz=fz,fselw=fselw,fz_grad=fz_grad,fz2_grad=fz2_grad)

def gen_softmax_data(filename='data.bin',ni=500,no=10,batch_size=384):
    t0 = time.time()
    random.seed()
    para = [ni,no,batch_size]
    b = [random.random()*2.-1. for i2 in xrange(no)]
    x = [[random.random()*2.-1. for i3 in xrange(batch_size)] for i1 in xrange(ni)]
    w = [[random.random()*2.-1. for i2 in xrange(no)] for i1 in xrange(ni)]
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

def gen_all_data():
    dns = map(lambda x:os.getenv('CNN_TEST_DATA_PATH')+x,
        ['/conv_large/','/conv_small/','/max_pool_large/','/max_pool_small/','/softmax/']
    )
    for d in dns:
        shutil.rmtree(d)
        print "[INFO] rm %s" %(d)
    for d in dns:
        os.mkdir(d)
        print "[INFO] mkdir %s" %(d)

    gen_conv_data(filename=dns[0]+'data0.bin',ni=1,no=2,row=28,col=28,k=5,batch_size=384)
    gen_conv_data(filename=dns[0]+'data1.bin',ni=1,no=4,row=28,col=28,k=5,batch_size=384)
    gen_conv_data(filename=dns[0]+'data2.bin',ni=2,no=2,row=28,col=28,k=5,batch_size=384)
    gen_conv_data(filename=dns[0]+'data3.bin',ni=4,no=2,row=28,col=28,k=5,batch_size=384)
    gen_conv_data(filename=dns[0]+'data4.bin',ni=1,no=14,row=28,col=28,k=5,batch_size=384)
    gen_conv_data(filename=dns[0]+'data5.bin',ni=2,no=8,row=28,col=28,k=5,batch_size=384)
    gen_conv_data(filename=dns[0]+'data6.bin',ni=1,no=20,row=28,col=28,k=5,batch_size=384)
    gen_conv_data(filename=dns[0]+'data7.bin',ni=1,no=20,row=28,col=28,k=5,batch_size=384)
    
    gen_conv_data(filename=dns[1]+'data0.bin',ni=1,no=2,row=12,col=12,k=5,batch_size=384)
    gen_conv_data(filename=dns[1]+'data1.bin',ni=1,no=4,row=12,col=12,k=5,batch_size=384)
    gen_conv_data(filename=dns[1]+'data2.bin',ni=2,no=2,row=12,col=12,k=5,batch_size=384)
    gen_conv_data(filename=dns[1]+'data3.bin',ni=2,no=8,row=12,col=12,k=5,batch_size=384)
    gen_conv_data(filename=dns[1]+'data4.bin',ni=4,no=8,row=12,col=12,k=5,batch_size=384)
    gen_conv_data(filename=dns[1]+'data5.bin',ni=10,no=20,row=12,col=12,k=5,batch_size=384)
    gen_conv_data(filename=dns[1]+'data6.bin',ni=20,no=50,row=12,col=12,k=5,batch_size=384)
    gen_conv_data(filename=dns[1]+'data7.bin',ni=20,no=50,row=12,col=12,k=5,batch_size=384)
    
    gen_max_pool_data(filename=dns[2]+'data0.bin',no=1,row=24,col=24,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[2]+'data1.bin',no=2,row=24,col=24,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[2]+'data2.bin',no=3,row=24,col=24,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[2]+'data3.bin',no=5,row=24,col=24,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[2]+'data4.bin',no=8,row=24,col=24,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[2]+'data5.bin',no=10,row=24,col=24,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[2]+'data6.bin',no=20,row=24,col=24,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[2]+'data7.bin',no=20,row=24,col=24,k=2,batch_size=384)
    
    gen_max_pool_data(filename=dns[3]+'data0.bin',no=1,row=8,col=8,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[3]+'data1.bin',no=2,row=8,col=8,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[3]+'data2.bin',no=5,row=8,col=8,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[3]+'data3.bin',no=10,row=8,col=8,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[3]+'data4.bin',no=20,row=8,col=8,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[3]+'data5.bin',no=30,row=8,col=8,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[3]+'data6.bin',no=40,row=8,col=8,k=2,batch_size=384)
    gen_max_pool_data(filename=dns[3]+'data7.bin',no=50,row=8,col=8,k=2,batch_size=384)

    gen_softmax_data(filename=dns[4]+'data0.bin',ni=1,no=10,batch_size=384)
    gen_softmax_data(filename=dns[4]+'data0.bin',ni=2,no=10,batch_size=384)
    gen_softmax_data(filename=dns[4]+'data0.bin',ni=5,no=10,batch_size=384)
    gen_softmax_data(filename=dns[4]+'data0.bin',ni=10,no=10,batch_size=384)
    gen_softmax_data(filename=dns[4]+'data0.bin',ni=50,no=10,batch_size=384)
    gen_softmax_data(filename=dns[4]+'data0.bin',ni=200,no=10,batch_size=384)
    gen_softmax_data(filename=dns[4]+'data0.bin',ni=500,no=10,batch_size=384)
    gen_softmax_data(filename=dns[4]+'data0.bin',ni=500,no=10,batch_size=384)

if __name__=='__main__':
    gen_all_data()



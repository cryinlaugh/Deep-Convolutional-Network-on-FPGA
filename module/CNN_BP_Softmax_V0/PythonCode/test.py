import os
import shutil
import random
import cPickle
import time

PROJ_NAME = 'CNN_BP_Softmax_V0'

def del_slic():
    fns = ['%s.py' %(PROJ_NAME),'_%s.so' %(PROJ_NAME)]
    dns = ['simutils']
    for fn in fns:
        os.remove(fn)
    for dn in dns:
        shutil.rmtree(dn)
    
def gen_slic(ver='Simulation'):
    os.system('sliccompile -t python -m ../RunRules/%s/maxfiles/%s.max' %(ver,PROJ_NAME))

import math
def gen_data(filename='data.bin',dim_1=500,dim_2=384,dim_3=10):
    t0 = time.time()
    para = [dim_1,dim_2,dim_3]
    random.seed()
    w = [[random.random() for i3 in xrange(dim_3)] for i1 in xrange(dim_1)]
    x = [[random.random() for i2 in xrange(dim_2)] for i1 in xrange(dim_1)]
    z = [[sum([x[i1][i2]*w[i1][i3] for i1 in xrange(dim_1)]) for i3 in xrange(dim_3)] for i2 in xrange(dim_2)]
    m = [max(z[i2]) for i2 in xrange(dim_2)]
    sb = [[math.exp(z[i2][i3]-m[i2]) for i3 in xrange(dim_3)] for i2 in xrange(dim_2)]
    al = [sum(sb[i2]) for i2 in xrange(dim_2)]
    sm = [[sb[i2][i3]/al[i2] for i3 in xrange(dim_3)] for i2 in xrange(dim_2)]
    pred = [max([(v,i3) for i3,v in enumerate(z[i2])],key=lambda x:x[0])[1] for i2 in xrange(dim_2)]
    std = [random.randint(0,dim_3-1) for i2 in xrange(dim_2)]
    delta = [[1. if i3==std[i2] else 0. for i3 in xrange(dim_3)] for i2 in xrange(dim_2)]
    smmd = [[sm[i2][i3]-delta[i2][i3] for i3 in xrange(dim_3)] for i2 in xrange(dim_2)]
    w_grad = [[[x[i1][i2]*(sm[i2][i3]-delta[i2][i3]) for i3 in xrange(dim_3)] for i2 in xrange(dim_2)] for i1 in xrange(dim_1)]
    b_grad = [[sm[i2][i3]-delta[i2][i3] for i3 in xrange(dim_3)] for i2 in xrange(dim_2)]
    z_grad = [[(sm[i2][i3]-delta[i2][i3])/float(dim_2) for i3 in xrange(dim_3)] for i2 in xrange(dim_2)]
    x_grad = [[sum([w[i1][i3]*z_grad[i2][i3] for i3 in xrange(dim_3)]) for i2 in xrange(dim_2)] for i1 in xrange(dim_1)]
    with open(filename,'w') as ouf:
        cPickle.dump(para,ouf)
        cPickle.dump(w,ouf)
        cPickle.dump(x,ouf)
        cPickle.dump(sm,ouf)
        cPickle.dump(std,ouf)
        cPickle.dump(w_grad,ouf)
        cPickle.dump(b_grad,ouf)
        cPickle.dump(x_grad,ouf)
        opt = (z,m,sb,al,pred,delta,smmd,z_grad)
        cPickle.dump(opt,ouf)
    print '[INFO] time used = %f' %(time.time()-t0)
    return para,w,x,sm,std,w_grad,b_grad,x_grad

def fix(x,s):
    l0 = len(x)
    l1 = (l0+s-1)/s*s
    if l0!=l1:
        print 'fix length from %d to %d' %(l0,l1)
        return x+[0.0]*(l1-l0)
    return x

type_size = 8

def check(name,eps,res,f):
    print 'len(res_%s) = %d ; %d = len(std_%s)' %(name,len(res),len(f),name)
    print '# diff >= %.15f : %d / %d' %(eps,len([1 for i in xrange(len(res)) if abs(res[i]-f[i])>=eps]),len(res))
    print 'max diff = %.15f' %(max([abs(res[i]-f[i]) for i in xrange(len(res))]))

def test(filename='data.bin'):
    t0 = time.time()
    import CNN_BP_Softmax_V0 as cnn
    print '[INFO] loading data : %s' %(filename)
    with open(filename) as inf:
        para = cPickle.load(inf)
        w = cPickle.load(inf)
        x = cPickle.load(inf)
        sm = cPickle.load(inf)
        std = cPickle.load(inf)
        w_grad = cPickle.load(inf)
        b_grad = cPickle.load(inf)
        x_grad = cPickle.load(inf)
        opt = cPickle.load(inf)
    dim_1,dim_2,dim_3 = para
    fw = [w[i1][i3] for i1 in xrange(dim_1) for i3 in xrange(dim_3)]
    fx = [x[i1][i2] for i1 in xrange(dim_1) for i2 in xrange(dim_2)]
    fsm = [sm[i2][i3] for i2 in xrange(dim_2) for i3 in xrange(dim_3)]
    fstd = std
    fw_grad = [w_grad[i1][i2][i3] for i1 in xrange(dim_1) for i2 in xrange(dim_2) for i3 in xrange(dim_3)]
    fb_grad = [b_grad[i2][i3] for i2 in xrange(dim_2) for i3 in xrange(dim_3)]
    fx_grad = [x_grad[i1][i2] for i1 in xrange(dim_1) for i2 in xrange(dim_2)]
    print '[INFO] Done: load data'
    print '[INFO] time used = %f' %(time.time()-t0)

    vec_size = 2
    dim_2_vec = dim_2/vec_size
    ticks = dim_1*dim_2_vec
    x_offset = 0
    x_size = dim_1*dim_2*type_size
    w_size = dim_1*dim_3*type_size
    sm_offset = x_size
    sm_size = dim_2*dim_3*type_size
    std_size = dim_2*4
    w_grad_size = dim_1*dim_2*dim_3*type_size
    b_grad_size = dim_2*dim_3*type_size
    x_grad_offset = sm_offset+sm_size
    x_grad_size = dim_1*dim_2*type_size

    print '[INFO] running writeLMem'
    cnn.CNN_BP_Softmax_V0_writeLMem(
        param_offset = x_offset,
        param_size = x_size,
        instream_cpu_to_lmem_at_cpu = fx
    )
    print '[INFO] running writeLMem'
    cnn.CNN_BP_Softmax_V0_writeLMem(
        param_offset = sm_offset,
        param_size = sm_size,
        instream_cpu_to_lmem_at_cpu = fsm
    )
    print '[INFO] Done: writeLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running kernel'
    res_b_grad,res_w_grad = cnn.CNN_BP_Softmax_V0(
        param_b_grad_size = b_grad_size,
        param_dim_1 = dim_1,
        param_softmax_offset = sm_offset,
        param_softmax_size = sm_size,
        param_std_size = std_size,
        param_ticks = ticks,
        param_w_grad_size = w_grad_size,
        param_w_size = w_size,
        param_x_grad_offset = x_grad_offset,
        param_x_grad_size = x_grad_size,
        param_x_offset = x_offset,
        param_x_size = x_size,
        instream_std = fstd,
        instream_w = fw
    )
    print '[INFO] Done: Conv'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMem'
    res_x_grad = cnn.CNN_BP_Softmax_V0_readLMem(
        param_offset = x_grad_offset,
        param_size = x_grad_size
    )
    print '[INFO] Done: readLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] checking'
    check('w_grad',1e-9,res_w_grad,fw_grad)
    check('b_grad',1e-9,res_b_grad,fb_grad)
    check('x_grad',1e-9,res_x_grad,fx_grad)
    print '[INFO] Done: check'

def main_0(ver='Simulation'):
    try:
        del_slic()
    except Exception as e:
        pass
    gen_slic(ver)
    gen_data()
    test()

def main_1():
    test()

if __name__=='__main__':
    main_1()


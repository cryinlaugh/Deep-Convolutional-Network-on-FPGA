import os
import shutil
import random
import cPickle
import time

PROJ_NAME = 'CNN_FW_Conv_V0'

def del_slic():
    fns = ['%s.py' %(PROJ_NAME),'_%s.so' %(PROJ_NAME)]
    dns = ['simutils']
    for fn in fns:
        os.remove(fn)
    for dn in dns:
        shutil.rmtree(dn)
    
def gen_slic(ver='Simulation'):
    os.system('sliccompile -t python -m ../RunRules/%s/maxfiles/%s.max' %(ver,PROJ_NAME))

def gen_data(filename='data.bin',ni=20,no=50,k=5,row=12,col=12,batch_size=384):
    t0 = time.time()
    para = [ni,no,row,col,k,batch_size]
    random.seed()
    b = [random.random() for i2 in xrange(no)]
    w = [[[[random.random() for i6 in xrange(k)] for i5 in xrange(k)] for i2 in xrange(no)] for i1 in xrange(ni)]
    x = [[[[random.random() for i7 in xrange(batch_size)] for i4 in xrange(col)] for i3 in xrange(row)] for i1 in xrange(ni)]
    z = [[[[sum([w[i1][i2][i5][i6]*x[i1][i3+i5][i4+i6][i7] for i1 in xrange(ni) for i5 in xrange(k) for i6 in xrange(k)])+b[i2] for i7 in xrange(batch_size)] for i4 in xrange(col-k+1)] for i3 in xrange(row-k+1)] for i2 in xrange(no)]
    with open(filename,'w') as ouf:
        cPickle.dump(para,ouf)
        cPickle.dump(b,ouf)
        cPickle.dump(w,ouf)
        cPickle.dump(x,ouf)
        cPickle.dump(z,ouf)
    print '[INFO] time used = %f' %(time.time()-t0)
    return para,b,w,x,z

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
    import CNN_FW_Conv_V0 as cnn
    print '[INFO] loading data : %s' %(filename)
    with open(filename) as inf:
        para = cPickle.load(inf)
        b = cPickle.load(inf)
        w = cPickle.load(inf)
        x = cPickle.load(inf)
        z = cPickle.load(inf)
    ni,no,row,col,k,batch_size = para
    fb = [b[i2] for i2 in xrange(no) for i3 in xrange(row-k+1) for i4 in xrange(col-k+1) for i7 in xrange(batch_size)]
    fw = [w[i1][i2][i5][i6] for i1 in xrange(ni) for i2 in xrange(no) for i5 in xrange(k) for i6 in xrange(k)]
    fx = [x[i1][i3][i4][i7] for i1 in xrange(ni) for i3 in xrange(row) for i4 in xrange(col) for i7 in xrange(batch_size)]
    fz = [z[i2][i3][i4][i7] for i2 in xrange(no) for i3 in xrange(row-k+1) for i4 in xrange(col-k+1) for i7 in xrange(batch_size)]

    ticks = no*row*col*batch_size
    x_offset = 0
    x_size_iter = row*col*batch_size*type_size
    x_size = ni*row*col*batch_size*type_size
    w_cnt_iter = no*k*k
    w_size_iter = no*k*k*type_size
    z_offset = x_size
    z_size = no*(row-k+1)*(col-k+1)*batch_size*type_size
    print '[INFO] Done: load data'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running writeLMem'
    cnn.CNN_FW_Conv_V0_writeLMem(
        param_offset = x_offset,
        param_size = x_size,
        instream_cpu_to_lmem_at_cpu = fx
    )
    cnn.CNN_FW_Conv_V0_writeLMem(
        param_offset = z_offset,
        param_size = z_size,
        instream_cpu_to_lmem_at_cpu = fb
    )
    print '[INFO] Done: writeLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    for st in xrange(ni):
        print '[INFO] running Conv ',st
        cnn.CNN_FW_Conv_V0(
            param_no = no,
            param_ticks = ticks,
            param_w_size = w_size_iter,
            param_x_offset = x_offset+st*x_size_iter,
            param_x_size = x_size_iter,
            param_z2_offset = z_offset,
            param_z2_size = z_size,
            instream_w = fw[st*w_cnt_iter:(st+1)*w_cnt_iter]
        )
        print '[INFO] Done: Conv'
        print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMem'
    res_z = cnn.CNN_FW_Conv_V0_readLMem(
        param_offset = z_offset,
        param_size = z_size
    )
    print '[INFO] Done: readLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] checking'
    check('z',1e-12,res_z,fz)
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


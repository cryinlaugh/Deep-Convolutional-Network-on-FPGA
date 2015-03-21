import os
import shutil
import random
import cPickle
import time

PROJ_NAME = 'CNN_FW_MaxPool_V0'

def del_slic():
    fns = ['%s.py' %(PROJ_NAME),'_%s.so' %(PROJ_NAME)]
    dns = ['simutils']
    for fn in fns:
        os.remove(fn)
    for dn in dns:
        shutil.rmtree(dn)
    
def gen_slic(ver='Simulation'):
    os.system('sliccompile -t python -m ../RunRules/%s/maxfiles/%s.max' %(ver,PROJ_NAME))

def gen_data(filename='data.bin',no=50,k=2,row=8,col=8,batch_size=384):
    t0 = time.time()
    para = [no,row,col,k,batch_size]
    random.seed()
    z2 = [[[[float(random.randint(0,10000)) for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    z = [[[[max([z2[i1][i2*k+i5][i3*k+i6][i4] for i5 in xrange(k) for i6 in xrange(k)]) for i4 in xrange(batch_size)] for i3 in xrange(col/k)] for i2 in xrange(row/k)] for i1 in xrange(no)]
    sels = [[[[1 if z2[i1][i2][i3][i4]==z[i1][i2/k][i3/k][i4] else 0 for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    sel = [[[[
        [sels[i1][i2+i5][i3+i6][i4] for i5 in xrange(k) for i6 in xrange(k)]
        for i4 in xrange(batch_size)] for i3 in xrange(0,col,k)] for i2 in xrange(0,row,k)] for i1 in xrange(no)]
    with open(filename,'w') as ouf:
        cPickle.dump(para,ouf)
        cPickle.dump(z2,ouf)
        cPickle.dump(z,ouf)
        cPickle.dump(sel,ouf)
    print '[INFO] time used = %f' %(time.time()-t0)
    return para,z2,z,sel

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
    import CNN_FW_MaxPool_V0 as cnn
    print '[INFO] loading data : %s' %(filename)
    with open(filename) as inf:
        para = cPickle.load(inf)
        z2 = cPickle.load(inf)
        z = cPickle.load(inf)
        sel = cPickle.load(inf)
    no,row,col,k,batch_size = para
    fz2 = [z2[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row) for i3 in xrange(col) for i4 in xrange(batch_size)]
    fz = [z[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row/k) for i3 in xrange(col/k) for i4 in xrange(batch_size)]
    fsels = [sel[i1][i2][i3][i4][i5] for i1 in xrange(no) for i2 in xrange(row/k) for i3 in xrange(col/k) for i4 in xrange(batch_size) for i5 in xrange(k*k)]
    ticks = no*row*col*batch_size
    z2_offset = 0
    z2_size = no*row*col*batch_size*type_size
    z_offset = z2_size
    z_size = no*(row/k)*(col/k)*batch_size*type_size
    sel_offset = z_offset+z_size
    sel_size = no*row*col*batch_size/8
    print '[INFO] Done: load data'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running writeLMem'
    cnn.CNN_FW_MaxPool_V0_writeLMem(
        param_offset = z2_offset,
        param_size = z2_size,
        instream_cpu_to_lmem_at_cpu = fz2
    )
    print '[INFO] Done: writeLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running Conv'
    cnn.CNN_FW_MaxPool_V0(
        param_no = no,
        param_sel_offset = sel_offset,
        param_sel_size = sel_size,
        param_ticks = ticks,
        param_z2_offset = z2_offset,
        param_z2_size = z2_size,
        param_z_offset = z_offset,
        param_z_size = z_size
    )
    print '[INFO] Done: Conv'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMem'
    res_z = cnn.CNN_FW_MaxPool_V0_readLMem(
        param_offset = z_offset,
        param_size = z_size
    )
    print '[INFO] Done: readLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMemBytes'
    res_sel = cnn.CNN_FW_MaxPool_V0_readLMemBytes(
        param_offset = sel_offset,
        param_size = sel_size
    )
    res_sels = [(x>>b)&1 for x in res_sel for b in xrange(8)]
    print '[INFO] Done: readLMemBytes'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] checking'
    check('z',1e-12,res_z,fz)
    check('sel',1e-12,res_sels,fsels)
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


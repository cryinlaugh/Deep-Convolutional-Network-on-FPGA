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
    #y2 = [[[[random.random() for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    y2 = [[[[float(random.randint(0,100)) for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    #y2 = [[[[((i1*row+i2)*col+i3)*batch_size+i4 for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    y = [[[[max([y2[i1][i2*k+i5][i3*k+i6][i4] for i5 in xrange(k) for i6 in xrange(k)]) for i4 in xrange(batch_size)] for i3 in xrange(col/k)] for i2 in xrange(row/k)] for i1 in xrange(no)]
    sels = [[[[1 if y2[i1][i2][i3][i4]==y[i1][i2/k][i3/k][i4] else 0 for i4 in xrange(batch_size)] for i3 in xrange(col)] for i2 in xrange(row)] for i1 in xrange(no)]
    sel = [[[[
        #sum(map(lambda x,y:x*y,[sels[i1][i2+i5][i3+i6][i4] for i5 in xrange(k) for i6 in xrange(k)],[1<<i for i in xrange(k*k)])) 
        [sels[i1][i2+i5][i3+i6][i4] for i5 in xrange(k) for i6 in xrange(k)]
        for i4 in xrange(batch_size)] for i3 in xrange(0,col,k)] for i2 in xrange(0,row,k)] for i1 in xrange(no)]
    with open(filename,'w') as ouf:
        cPickle.dump(para,ouf)
        cPickle.dump(y2,ouf)
        cPickle.dump(y,ouf)
        cPickle.dump(sel,ouf)
    print '[INFO] time used = %f' %(time.time()-t0)
    return para,y2,y,sel

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
        y2 = cPickle.load(inf)
        y = cPickle.load(inf)
        sel = cPickle.load(inf)
    no,row,col,k,batch_size = para
    fy2 = [y2[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row) for i3 in xrange(col) for i4 in xrange(batch_size)]
    fy = [y[i1][i2][i3][i4] for i1 in xrange(no) for i2 in xrange(row/k) for i3 in xrange(col/k) for i4 in xrange(batch_size)]
    fsels = [sel[i1][i2][i3][i4][i5] for i1 in xrange(no) for i2 in xrange(row/k) for i3 in xrange(col/k) for i4 in xrange(batch_size) for i5 in xrange(k*k)]
    print '[INFO] Done: load data'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running writeLMem'
    cnn.CNN_FW_MaxPool_V0_writeLMem(
        param_offset = 0,
        param_size = len(fy2)*type_size,
        instream_cpu_to_lmem_at_cpu = fy2
    )
    print '[INFO] Done: writeLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running Conv'
    cnn.CNN_FW_MaxPool_V0(
        param_no = no,
        param_sel_offset = (len(fy2)+len(fy))*type_size,
        param_sel_size = len(fsels)/8,
        param_ticks = no*row*col*batch_size,
        param_y2_offset = 0,
        param_y2_size = len(fy2)*type_size,
        param_y_offset = len(fy2)*type_size,
        param_y_size = len(fy)*type_size
    )
    print '[INFO] Done: Conv'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMem'
    res_y = cnn.CNN_FW_MaxPool_V0_readLMem(
        param_offset = len(fy2)*type_size,
        param_size = len(fy)*type_size
    )
    print '[INFO] Done: readLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMemBytes'
    res_sel = cnn.CNN_FW_MaxPool_V0_readLMemBytes(
        param_offset = (len(fy2)+len(fy))*type_size,
        param_size = len(fsels)/8
    )
    tmp = [[(x>>b)&1 for b in xrange(8)] for x in res_sel]
    res_sels = [tmp[i][j] for i in xrange(len(res_sel)) for j in xrange(8)]
    print '[INFO] Done: readLMemBytes'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] checking'
    check('y',1e-12,res_y,fy)
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


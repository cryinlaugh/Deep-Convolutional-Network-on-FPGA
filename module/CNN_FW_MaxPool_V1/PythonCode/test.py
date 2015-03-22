import sys
import os
sys.path.append(os.getenv('CNN_TEST_UTIL_PATH'))
import time
import cPickle
import utils
utils.PROJ_NAME = 'CNN_FW_MaxPool_V1'
TEST_DATA_PATH=os.getenv('CNN_TEST_DATA_PATH')+'/max_pool_small'
TEST_DATA_SET = map(lambda f:os.path.join(TEST_DATA_PATH,f),os.listdir(TEST_DATA_PATH))
type_size = 8

def test(filename='data.bin'):
    t0 = time.time()
    import CNN_FW_MaxPool_V1 as cnn
    print '[INFO] loading data : %s' %(filename)
    with open(filename) as inf:
        para = cPickle.load(inf)
        fz2 = cPickle.load(inf)
        fz = cPickle.load(inf)
        fselw = cPickle.load(inf)
    no,row,col,k,batch_size = para
    
    z2_offset = 0
    z2_size = no*row*col*batch_size*type_size
    z_offset = z2_size
    z_size = no*(row/k)*(col/k)*batch_size*type_size
    sel_offset = z_offset+z_size
    sel_size = no*row*col*batch_size/8
    print '[INFO] Done: load data'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running writeLMem'
    cnn.CNN_FW_MaxPool_V1_writeLMem(
        param_offset = z2_offset,
        param_size = z2_size,
        instream_cpu_to_lmem_at_cpu = fz2
    )
    print '[INFO] Done: writeLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running Conv'
    cnn.CNN_FW_MaxPool_V1(
        param_no = no,
        param_sel_offset = sel_offset,
        param_z2_offset = z2_offset,
        param_z_offset = z_offset
    )
    print '[INFO] Done: Conv'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMem'
    res_z = cnn.CNN_FW_MaxPool_V1_readLMem(
        param_offset = z_offset,
        param_size = z_size
    )
    print '[INFO] Done: readLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMemBytes'
    res_selw = cnn.CNN_FW_MaxPool_V1_readLMemBytes(
        param_offset = sel_offset,
        param_size = sel_size
    )
    #res_sels = [(x>>b)&1 for x in res_sel for b in xrange(8)]
    print '[INFO] Done: readLMemBytes'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] checking'
    ret1 = utils.check('z',1e-12,res_z,fz)
    ret2 = utils.check('selw',0,res_selw,fselw)
    print '[INFO] Done: check'
    return ret1 and ret2

def main_1(ver='Simulation'):
    try:
        utils.del_slic()
    except Exception as e:
        pass
    utils.gen_slic(ver)
def main_2(filename=None):
    if filename is None:
        fns = TEST_DATA_SET
    else:
        fns = [filename]
    fns.sort()
    for f in fns:
        if not test(f):
            return False
    return True
def main_0():
    main_1()
    main_2()

if __name__=='__main__':
    main_0()


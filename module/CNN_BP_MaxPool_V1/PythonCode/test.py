import sys
import os
sys.path.append(os.getenv('CNN_TEST_UTIL_PATH'))
import time
import cPickle
import utils
utils.PROJ_NAME = 'CNN_BP_MaxPool_V1'
TEST_DATA_PATH = os.getenv('CNN_TEST_DATA_PATH')+'/max_pool_small'
TEST_DATA_SET = map(lambda f:os.path.join(TEST_DATA_PATH,f),os.listdir(TEST_DATA_PATH))
type_size = 8

def test(filename='data.bin'):
    t0 = time.time()
    import CNN_BP_MaxPool_V1 as cnn
    print '[INFO] loading data : %s' %(filename)
    with open(filename) as inf:
        para = cPickle.load(inf)
        fz2 = cPickle.load(inf)
        fselw = cPickle.load(inf)
        fz = cPickle.load(inf)
        fa = cPickle.load(inf)
        fa_grad = cPickle.load(inf)
        fz_grad = cPickle.load(inf)
        fz2_grad = cPickle.load(inf)
    no,row,col,k,batch_size = para
    
    sel_offset = 0
    sel_size = no*row*col*batch_size/8
    z_offset = sel_size
    z_size = no*(row/k)*(col/k)*batch_size*type_size
    a_grad_offset = z_offset+z_size
    a_grad_size = no*(row/k)*(col/k)*batch_size*type_size
    z2_grad_offset = a_grad_offset+a_grad_size
    z2_grad_size = no*row*col*batch_size*type_size
    print '[INFO] Done: load data'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running writeLMem'
    cnn.CNN_BP_MaxPool_V1_writeLMem(
        param_offset = z_offset,
        param_size = z_size,
        instream_cpu_to_lmem_at_cpu = fz
    )
    print '[INFO] Done: writeLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running writeLMem'
    cnn.CNN_BP_MaxPool_V1_writeLMem(
        param_offset = a_grad_offset,
        param_size = a_grad_size,
        instream_cpu_to_lmem_at_cpu = fa_grad
    )
    print '[INFO] Done: writeLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running writeLMemBytes'
    cnn.CNN_BP_MaxPool_V1_writeLMemBytes(
        param_offset = sel_offset,
        param_size = sel_size,
        instream_cpu_to_lmem_at_cpu = fselw
    )
    print '[INFO] Done: readLMemBytes'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running Conv'
    cnn.CNN_BP_MaxPool_V1(
        param_a_grad_offset = a_grad_offset,
        param_no = no,
        param_sel_offset = sel_offset,
        param_z2_grad_offset = z2_grad_offset,
        param_z_offset = z_offset
    )
    print '[INFO] Done: Conv'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMem'
    res_z2_grad = cnn.CNN_BP_MaxPool_V1_readLMem(
        param_offset = z2_grad_offset,
        param_size = z2_grad_size
    )
    print '[INFO] Done: readLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] checking'
    ret = utils.check('z2_grad',1e-12,res_z2_grad,fz2_grad)
    print '[INFO] Done: check'
    return ret

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
def main_0(ver='Simulation'):
    main_1()
    main_2()

if __name__=='__main__':
    main_0()


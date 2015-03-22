import sys
import os
sys.path.append(os.getenv('CNN_TEST_UTIL_PATH'))
import time
import cPickle
import utils
utils.PROJ_NAME = 'CNN_FW_Conv_V1'
TEST_DATA_PATH=os.getenv('CNN_TEST_DATA_PATH')+'/conv_small'
TEST_DATA_SET = map(lambda f:os.path.join(TEST_DATA_PATH,f),os.listdir(TEST_DATA_PATH))
type_size = 8

def test(filename='data.bin'):
    t0 = time.time()
    import CNN_FW_Conv_V1 as cnn
    print '[INFO] loading data : %s' %(filename)
    with open(filename) as inf:
        para = cPickle.load(inf)
        fb = cPickle.load(inf)
        fw = cPickle.load(inf)
        fx = cPickle.load(inf)
        fz = cPickle.load(inf)
    ni,no,row,col,k,batch_size = para

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
    cnn.CNN_FW_Conv_V1_writeLMem(
        param_offset = x_offset,
        param_size = x_size,
        instream_cpu_to_lmem_at_cpu = fx
    )
    cnn.CNN_FW_Conv_V1_writeLMem(
        param_offset = z_offset,
        param_size = z_size,
        instream_cpu_to_lmem_at_cpu = fb
    )
    print '[INFO] Done: writeLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    for st in xrange(ni):
        print '[INFO] running Conv ',st
        cnn.CNN_FW_Conv_V1(
            param_no = no,
            param_x_offset = x_offset+st*x_size_iter,
            param_z_offset = z_offset,
            instream_w = fw[st*w_cnt_iter:(st+1)*w_cnt_iter]
        )
        print '[INFO] Done: Conv'
        print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMem'
    res_z = cnn.CNN_FW_Conv_V1_readLMem(
        param_offset = z_offset,
        param_size = z_size
    )
    print '[INFO] Done: readLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] checking'
    ret = utils.check('z',1e-12,res_z,fz)
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
def main_0():
    main_1()
    main_2()

if __name__=='__main__':
    main_0()


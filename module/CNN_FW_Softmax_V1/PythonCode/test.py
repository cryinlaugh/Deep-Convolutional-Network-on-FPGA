import sys
import os
sys.path.append(os.getenv('CNN_TEST_UTIL_PATH'))
import time
import cPickle
import utils
utils.PROJ_NAME = 'CNN_FW_Softmax_V1'
TEST_DATA_PATH = os.path.join(os.getenv('CNN_TEST_DATA_PATH'),'softmax')
TEST_DATA_SET = map(lambda f:os.path.join(TEST_DATA_PATH,f),os.listdir(TEST_DATA_PATH))
type_size = 8

def test(filename='data.bin'):
    t0 = time.time()
    import CNN_FW_Softmax_V1 as cnn
    print '[INFO] loading data : %s' %(filename)

    with open(filename) as inf:
        para = cPickle.load(inf)
        fb = cPickle.load(inf)
        fx = cPickle.load(inf)
        fw = cPickle.load(inf)
        fz = cPickle.load(inf)
        fmaxz = cPickle.load(inf)
        fexpz = cPickle.load(inf)
        fsumexpz = cPickle.load(inf)
        fsm = cPickle.load(inf)
        fpred = cPickle.load(inf)
        fstd = cPickle.load(inf)
        fb_grad = cPickle.load(inf)
        fw_grad = cPickle.load(inf)
        fx_grad = cPickle.load(inf)
    ni,no,batch_size = para

    x_offset = 0
    x_size = ni*batch_size*type_size
    sm_offset = x_size
    sm_size = no*batch_size*type_size
    print '[INFO] Done: load data'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running writeLMem'
    cnn.CNN_FW_Softmax_V1_writeLMem(
        param_offset = x_offset,
        param_size = x_size,
        instream_cpu_to_lmem_at_cpu = fx
    )
    print '[INFO] Done: writeLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running kernel'
    res_pred = cnn.CNN_FW_Softmax_V1(
        param_ni = ni,
        param_x_offset = x_offset,
        param_softmax_offset = sm_offset,
        instream_b = fb,
        instream_w = fw
    )
    print '[INFO] Done: Conv'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMem'
    res_sm = cnn.CNN_FW_Softmax_V1_readLMem(
        param_offset = sm_offset,
        param_size = sm_size
    )
    print '[INFO] Done: readLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] checking'
    ret0 = utils.check('sm',1e-9,res_sm,fsm)
    ret1 = utils.check('pred',1e-9,res_pred,fpred)
    print '[INFO] Done: check'
    return ret0 and ret1

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
    main_1(ver)
    main_2()

if __name__=='__main__':
    main_0()


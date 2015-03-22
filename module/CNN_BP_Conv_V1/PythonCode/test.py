import sys
import os
sys.path.append(os.getenv('CNN_TEST_UTIL_PATH'))
import time
import cPickle
import utils
utils.PROJ_NAME = 'CNN_BP_Conv_V1'
TEST_DATA_PATH=os.getenv('CNN_TEST_DATA_PATH')
TEST_DATA_SET = map(lambda f:os.path.join(TEST_DATA_PATH+'/conv_small',f),os.listdir(TEST_DATA_PATH+'/conv_small'))
type_size = 8

def test(filename='data.bin'):
    t0 = time.time()
    import CNN_BP_Conv_V1 as cnn
    print '[INFO] loading data : %s' %(filename)
    with open(filename) as inf:
        para = cPickle.load(inf)
        fb = cPickle.load(inf)
        fw = cPickle.load(inf)
        fx = cPickle.load(inf)
        fz = cPickle.load(inf)
        fz_grad = cPickle.load(inf)
        fx_grad = cPickle.load(inf)
        fw_grad = cPickle.load(inf)
    ni,no,row,col,k,batch_size = para

    x_offset = 0
    x_size_iter = row*col*batch_size*type_size
    x_size = ni*row*col*batch_size*type_size
    w_cnt_iter = no*k*k
    z_grad_offset = x_size
    z_grad_size = no*(row-k+1)*(col-k+1)*batch_size*type_size
    x_grad_old_offset = z_grad_offset+z_grad_size
    x_grad_old_size_iter = row*col*batch_size*type_size
    x_grad_old_size = ni*row*col*batch_size*type_size
    x_grad_old_cnt = ni*row*col*batch_size
    x_grad_offset = x_grad_old_offset
    x_grad_size_iter = x_grad_old_size_iter
    x_grad_size = x_grad_old_size
    print '[INFO] Done: load data'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running writeLMem'
    cnn.CNN_BP_Conv_V1_writeLMem(
        param_offset = z_grad_offset,
        param_size = z_grad_size,
        instream_cpu_to_lmem_at_cpu = fz_grad
    )
    print '[INFO] running writeLMem'
    cnn.CNN_BP_Conv_V1_writeLMem(
        param_offset = x_offset,
        param_size = x_size,
        instream_cpu_to_lmem_at_cpu = fx
    )
    print '[INFO] running writeLMem'
    cnn.CNN_BP_Conv_V1_writeLMem(
        param_offset = x_grad_old_offset,
        param_size = x_grad_old_size,
        instream_cpu_to_lmem_at_cpu = [0.0]*x_grad_old_cnt
    )
    print '[INFO] Done: writeLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    res_w_grad = []
    for st in xrange(ni):
        print '[INFO] running Conv ',st
        tmp = cnn.CNN_BP_Conv_V1(
            param_no = no,
            param_x_grad_offset = x_grad_offset+x_grad_size_iter*st,
            param_x_grad_old_offset = x_grad_old_offset+x_grad_old_size_iter*st,
            param_x_offset = x_offset+x_size_iter*st,
            param_z_grad_offset = z_grad_offset,
            instream_w = fw[st*w_cnt_iter:(st+1)*w_cnt_iter]
        )
        #res_w_grad += [sum(tmp[i*batch_size:(i+1)*batch_size]) for i in xrange(no*k*k)]
        res_w_grad += tmp
        print '[INFO] Done: Conv'
        print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] running readLMem'
    res_x_grad = cnn.CNN_BP_Conv_V1_readLMem(
        param_offset = x_grad_offset,
        param_size = x_grad_size
    )
    print '[INFO] Done: readLMem'
    print '[INFO] time used = %f' %(time.time()-t0)

    print '[INFO] checking'
    ret1 = utils.check('x_grad',1e-12,res_x_grad,fx_grad)
    ret2 = utils.check('w_grad',1e-12,res_w_grad,fw_grad)
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


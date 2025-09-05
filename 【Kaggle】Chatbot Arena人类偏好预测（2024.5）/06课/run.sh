CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"  accelerate launch train.py
#accelerate launch  train.py
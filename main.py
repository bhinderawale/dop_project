from xml.parsers.expat import model
import torch
from quant import *
from outlier import *
from evaluate import *
from collections import defaultdict
from pprint import pprint
import argparse
from dataHelpers import generate_data, format_input
from quant import enable_dynamic_AQ, quantize_model_func
#from datautils import *
from SATVN import SATVN
from ASATVN import ASATVN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)

    parser.add_argument('--wbits', type=int, default=16, choices=[2,3,4,8,16])
    parser.add_argument('--abits', type=int, default=16, choices=[2,3,4,8,16])

    parser.add_argument('--exponential', action='store_true')
    parser.add_argument('--a_sym', action='store_true')
    parser.add_argument('--w_sym', action='store_true')

    parser.add_argument('--weight_group_size', type=int, default=0)
    parser.add_argument('--weight_channel_group', type=int, default=1)
    parser.add_argument('--act_group_size', type=int, default=0)

    parser.add_argument('--keeper', type=int, default=0)
    parser.add_argument('--keeper_precision', type=int, default=0)

    parser.add_argument('--tiling', type=int, default=0)
    parser.add_argument('--a_clip_ratio', type=float, default=1.0)
    parser.add_argument('--w_clip_ratio', type=float, default=1.0)

    parser.add_argument('--quant_type', type=str, default='int', choices=['int','fp'])

    parser.add_argument('--use_downsampling', action='store_true')
    parser.add_argument('--downsample_every', type=int, default=2)
    
    return parser.parse_args()    
    
def main():
    args = parse_args()
    model = ASATVN(args).to(device)
    print("initialising model")

    if args.abits < 16:
        print("enabling dynamic AQ")
        enable_dynamic_AQ(model, args)
    if args.wbits < 16:
        print("quantising weights")
        model = quantize_model_func(model, device, args)
    test_x = format_input(test_x)
    test_y = format_input(test_y)
    mase, smape, nrmse = evaluate(model, test_x, test_y)
    print(f"MASE: {mase}, SMAPE: {smape}, NRMSE: {nrmse}")

if __name__ == '__main__':
    main()
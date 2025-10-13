import argparse


def get_parser():

    parser = argparse.ArgumentParser(description='Hyperparameters for StoX-Net Training & Inference')

    ######################### #########################################################################################
    ## Regular Hyperparameters
    ##################################################################################################################
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128,type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (def5ault: 1e-4)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate (default: 0.01)')
    ##################################################################################################################
    ## High-level trainer parameters
    ##################################################################################################################
    parser.add_argument('--model', dest='model', default='resnet18', type=str, 
                        help='Choose a model to run the network on {resnet20, resnet18}')
    parser.add_argument('--dataset', dest='dataset', help='Choose a dataset to run the network on from'
                                                          '{MNIST, CIFAR10}', default='CIFAR10', type=str)
    parser.add_argument('--experiment_state', default='PTQAT', type=str,
                        help='What are we doing right now? options: [pretraining, pruning,PTQAT,inference]')
    parser.add_argument('--run-info', default='QAF18_raw', type=str,
                        help='Anything to add to the run name for clarification? e.g. \"test1ab\"')
   
    #parser.add_argument('--quantized', dest='quantized', default=False,pre+gf
    #                   type=bool, help='Select whether use the quantized conv layer or not') 
    ##################################################################################################################
    ## Saving/Loading Data
    ##################################################################################################################
    parser.add_argument('--checkpoint-path', default='', type=str, metavar='PATH',
                        help='absolute path to desired checkpoint (default: none)')
    parser.add_argument('--model-save-dir', dest='model_save_dir',
                        help='The directory used to save the trained models',
                        default='./saved/models/', type=str)
    parser.add_argument('--logs-save-dir', dest='logs_save_dir',
                        help='The directory used to save the trained models',
                        default='./saved/logs/', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', type=str, default='/home/weiweiz/Documents/WW_03/saved/models/_resnet18_CIFAR10_20_128_0p001_pruning_0_1_True_0_1_False_1_False_False.th')
#
    ##################################################################################################################
    ## QAT / Model Parameters
    ##################################################################################################################
    # xbar precision params
    parser.add_argument('--weight_bits', default=4, type=int, metavar='N',
                        help='Number of weight bits (default: 4)')
    parser.add_argument('--input_bits', default=4, type=int, metavar='N',
                        help='Number of input bits (default:4)')
    parser.add_argument('--weight_bit_frac', default=4, type=int, metavar='N',
                        help='Number of frac weight bits (default: 4)')
    parser.add_argument('--input_bit_frac', default=1, type=int, metavar='N',
                        help='Number of frac input bits (default: 1)')
    parser.add_argument('--bit_slice', default=1, type=int, metavar='N',
                        help='Number of weight bits per slice (default: 1), x <= 0 means full precision')
    
    parser.add_argument('--bit_stream', default=1, type=int, metavar='N',
                        help='Number of input bits per slice (default: 1), x <= 0 means full precision')
    parser.add_argument('--subarray-size', default=128, type=int, metavar='N',
                        help='Size of partial sum subarrays, x <= 0 means no partial sums')
    
    # adc params

    parser.add_argument('--Gon', default=1/10, type=int, metavar='N',
                        help='max conductance of ADC')
    parser.add_argument('--Goff', default=1/1000, type=int, metavar='N',
                        help='min conductance of ADC')
    parser.add_argument('--adc_bit', default=1, type=int, metavar='N',
                        help='ADC precision for quantized layers, x <= 0 means full precision')
    parser.add_argument('--save-adc', dest='save_adc', default=False,
                        type=bool, help='Select whether the ADC inputs are saved for analysis')
    parser.add_argument('--adc-grad-filter', dest='adc_grad_filter', action='store_true', default=False,
                    help='Select whether an STE (False) or halfsine (True) is used for ADC backprop')
    parser.add_argument('--acm_bits', dest='acm_bits', default=32,
                        type=bool, help=' ')
    parser.add_argument('--acm_bit_frac', dest='acm_bit_frac', default=24,
                        type=bool, help=' ')
    parser.add_argument('--calibrate_adc', dest='calibrate_adc', default=False, type=bool, 
                    help='Enable ADC calibration for PTQ') 
   
    parser.add_argument('--Vmax', default=1, type=int,    
                        help='ADC Votalge')                    
    parser.add_argument('--adc_custom_loss', action='store_true', default=False,
                    help='Enable ADC custom regularization loss')
    parser.add_argument('--adc_reg_lambda', type=float, default=0.1,
                        help='ADC regularization lambda coefficient')
    
    
    # other?
    parser.add_argument('--wa-stoch-round', dest='wa_stoch_round', default=True, 
                        help='Select whether stochaastic or deterministic rounding is used for ADC')
    parser.add_argument('--conv_prune_rate', default=0.7, type=float,
                        help='Set prune rate for')
    parser.add_argument('--linear_prune_rate', default=0.7, type=float,
                        help='Set prune rate for')
    parser.add_argument('--input_prune_rate', default=0.5, type=float,
                        help='Set prune rate for')
    parser.add_argument('--viz-comp-graph', default=False, type=bool, 
                        help='use torchviz to show model computational graph fwd/bkwd')
# Calibration parameters (add after existing ADC params)
    
    ##################################################################################################################
    ## VQ Parameters 
    ##################################################################################################################
    # In your argsparser.py, add these arguments:
    parser.add_argument('--use-vq', dest='use_vq', action='store_true', default=False,
                    help='Use Vector Quantization')
    parser.add_argument('--vq-loss-weight', default=0.01, type=float,
                    help='VQ loss weight')
   
    ##################################################################################################################
    ## Miscellaneous parameters
    ##################################################################################################################
    parser.add_argument('--print-batch-info', dest='print_batch_info',
                        help='Set to true if you want to see per batch accuracy',
                        default=False, type=bool)
    parser.add_argument('--skip-to-batch', default=0, type=int,
                        metavar='N', help='Skip to this batch of images in inference')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 20)')
    

    parser.add_argument('--adc-slice-weighting', type=str, default='exponential',
                    choices=['none', 'linear', 'exponential'],
                    help='Loss weighting for bit slices: ' +
                         'exponential=2^(slice*bits) (default), ' +
                         'linear=1+slice, none=uniform')
    parser.add_argument('--adc-slice-weight-scale', type=float, default=1.0,
                        help='Multiplier for slice weights (default: 1.0)')
    return parser

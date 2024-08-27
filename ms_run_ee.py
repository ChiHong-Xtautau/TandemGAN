import sys
import argparse

from ms_common import Logger
from ms_ee_attack import MSEEAttack
from ms_defense import MSDefense
import ms_common as comm


def run_ee(args, eval_S):

    if not args.l_only:
        scenario = "p_only"
    else:
        scenario = "l_only"
    if not eval_S:
        msd = MSDefense(args)

        if args.dataset == 'MNIST':
            msd.load(netv_path='saved_model/pretrained_net/vgg16_mnist.pth')
        elif args.dataset == 'FashionMNIST':
            msd.load(netv_path='saved_model/pretrained_net/vgg16_fashion_mnist.pth')
        elif args.dataset == 'CIFAR10':
            msd.load(netv_path='saved_model/pretrained_net/resnet34_3_cifar10.pth')
        elif args.dataset == 'SVHN':
            msd.load(netv_path='saved_model/pretrained_net/resnet34_svhn.pth')
        else:
            return

        msa = MSEEAttack(args, defense_obj=msd)
        msa.load()

        comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
        comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)

        msa.ee_train_netS('temp/alg/ee_netS_%s_%s' % (args.dataset, scenario),
                       'temp/alg/ee_netG_%s_%s' % (args.dataset, scenario), 'temp/alg/ee_netGE_%s_%s' % (args.dataset, scenario))

        comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
        comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)

    else:
        msd = MSDefense(args)

        msd.load(netv_path='saved_model/pretrained_net/resnet34_3_cifar10.pth')

        msa = MSEEAttack(args, defense_obj=msd)
        msa.load(nets_path='saved_model/dfme_trained_net/dfme_netS_CIFAR10_p_only_over.pth')

        print(args.dataset, ',', scenario)
        print("***** DFME *****")
        comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=args.cuda)
        comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=args.cuda)

        msa.attack("FGSM")
        msa.attack("BIM")
        msa.attack("PGD")
        msa.attack("FGSM", targeted=True)
        msa.attack("BIM", targeted=True)
        msa.attack("PGD", targeted=True)


def adv_att_exp(args):
    msd = MSDefense(args)

    msd.load(netv_path='saved_model/pretrained_net/resnet34_3_cifar10.pth')

    msa = MSEEAttack(args, defense_obj=msd)

    # fpath = 'saved_model/p_only_models/maze_netS_CIFAR10_p_only_epoch_188.pth'
    # fpath = 'saved_model/p_only_models/dfme_netS_CIFAR10_p_only_epoch_14.pth'
    # fpath = 'saved_model/p_only_models/netD_epoch_182.pth'
    fpath = 'saved_model/p_only_models/ee_netS_CIFAR10_p_only_over.pth'
    msa.load(nets_path=fpath)

    # print("cifar10", ',', "p_only")
    # print(fpath)

    comm.accuracy(msa.netS, 'netS', test_loader=msa.test_loader, cuda=True)
    comm.accuracy(msd.netV, 'netV', test_loader=msd.test_loader, cuda=True)

    msa.attack("FGSM")
    # msa.attack("BIM")
    # msa.attack("PGD")
    # msa.attack("FGSM", targeted=True)
    # msa.attack("BIM", targeted=True)
    # msa.attack("PGD", targeted=True)


def visualization_syn(args):
    msd = MSDefense(args)
    if not args.dataset == 'CIFAR10':
        return

    msd.load(netv_path='saved_model/pretrained_net/resnet34_3_cifar10.pth')

    alg = MSEEAttack(args, defense_obj=msd)
    alg.load()

    if args.dataset == "CIFAR10" and (not args.l_only):
        alg.visualization_syn(netg_path='temp/alg/cifar10/ee_netG_CIFAR10_p_only_over.pth', netge_path='temp/alg/cifar10/ee_netGE_CIFAR10_p_only_over.pth')


def visualization_adv(args):
    msd = MSDefense(args)
    if not args.dataset == 'CIFAR10':
        return

    alg = MSEEAttack(args, defense_obj=msd)

    if args.dataset == "CIFAR10" and (not args.l_only):
        alg.load(nets_path="temp/alg/cifar10/ee_netS_CIFAR10_p_only_over.pth")

    alg.visualization_adv("FGSM")
    alg.visualization_adv("BIM")
    alg.visualization_adv("PGD")


def get_args(dataset, cuda, expt="attack", l_only=False):
    args = argparse.ArgumentParser()

    args.add_argument('--cuda', default=cuda, action='store_true', help='using cuda')
    args.add_argument('--num_class', type=int, default=10)

    args.add_argument('--epoch_exploit', type=int, default=1, help='fix to be one')

    if not l_only:
        args.add_argument('--epoch_itrs', type=int, default=50)
        args.add_argument('--epoch_dg_s', type=int, default=5, help='for training dynamic net G and net S')
        args.add_argument('--epoch_dg_g', type=int, default=1, help='for training dynamic net G and net S')
        args.add_argument('--epoch_dg_ge', type=int, default=1, help='for training dynamic net G and net S')
    else:
        args.add_argument('--epoch_itrs', type=int, default=50)
        args.add_argument('--epoch_dg_s', type=int, default=5, help='for training dynamic net G and net S')
        args.add_argument('--epoch_dg_g', type=int, default=1, help='for training dynamic net G and net S')
        args.add_argument('--epoch_dg_ge', type=int, default=1, help='for training dynamic net G and net S')

    args.add_argument('--z_dim', type=int, default=128, help='the dimension of noise')
    args.add_argument('--batch_size_g', type=int, default=256, help='for training net G and net S')

    args.add_argument('--attack_exp', default=(expt == "attack"), action='store_true', help='running attack experiments')
    args.add_argument('--test_exp', default=(expt == "test"), action='store_true', help='running attack experiments')

    args.add_argument('--l_only', default=l_only, action='store_true', help='label only')

    if not l_only:
        args.add_argument('--epoch_dg', type=int, default=200, help='for training dynamic net G and net S')
        args.add_argument('--lr_tune_s', type=float, default=0.001)
        args.add_argument('--lr_tune_g', type=float, default=0.0001)
        # args.add_argument('--lr_tune_g', type=float, default=0.0)  # for w/o exploration
        args.add_argument('--lr_tune_ge', type=float, default=1e-6)
        # args.add_argument('--lr_tune_ge', type=float, default=0.0)
        args.add_argument('--steps', nargs='+', default=[0.8, 0.95], type=float)
        args.add_argument('--scale', type=float, default=3e-1)
    else:
        args.add_argument('--epoch_dg', type=int, default=200, help='for training dynamic net G and net S')
        args.add_argument('--lr_tune_s', type=float, default=0.001)
        args.add_argument('--lr_tune_g', type=float, default=0.0001)
        args.add_argument('--lr_tune_ge', type=float, default=0.000001)
        args.add_argument('--steps', nargs='+', default=[0.7, 0.8], type=float)
        args.add_argument('--scale', type=float, default=3e-1)
    args.add_argument('--dataset', type=str, default='CIFAR10')
    args.add_argument('--res_filename', type=str, default='cifar10_ee')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    sys.stdout = Logger('t/ee_normal.log', sys.stdout)

    # args = get_args(dataset="cifar10", cuda=True, expt="attack", l_only=True)
    args = get_args(dataset="cifar10", cuda=True, expt="attack", l_only=False)
    print(args)

    print("***********************")
    if args.attack_exp:
        print("Attack Experiments:")
        run_ee(args, eval_S=False)
    if args.test_exp:
        print("Test Attacks Experiments:")
        run_ee(args, eval_S=True)

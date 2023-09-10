import os
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import json

from utils.data_loader import get_dataloader
from utils.train_utils import train_one_epoch,evaluate,train_model_one_epoch_without_event,evaluate_model_without_event,ConfusionMatrix,to_np,get_param_number
from utils.plot_curve import plot_loss_acc_picture,plot_step_acc_pic
from models.MCAN import  MCAN
from models.Adapter import Adapter,GroupAdapter

def main(args):

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print(device)

    if not os.path.exists('./{}_result'.format(args.model_name)):
        os.mkdir('./{}_result'.format(args.model_name))
    # 用来保存训练以及验证过程中信息
    results_file = "./{}_result/seed_{}_epochs_{}_{}.txt".format(args.model_name,args.seed,args.epochs,args.datasets)

    #create tensorboard
    # print('Start Tensorboard with "tensorboard --logdir=EANN_result", view at http://localhost:6006/')
    # tb_writer = SummaryWriter(log_dir='./EANN_result')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.emb_type == 'w2v':
        args.embed_dim = args.w2v_emb_dim
        args.hidden_dim = args.w2v_emb_dim
        train_loader, valid_loader, W = get_dataloader(args)
    elif args.emb_type == 'bert':
        train_loader, valid_loader = get_dataloader(args)

    # print('building model')
    if args.use_Adapter:
        # adapter = Adapter(256,128)
        adapter = GroupAdapter(256,64,8)
        model = MCAN(args.bert, args, adapter)
    else:
        model = MCAN(args.bert, args)
    # model.cuda()
    model.to(device)
    #use mixgrad
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    print('args.amp: ',args.amp,' scaler: ',scaler)
    print('args.pretrain: ',args.pretrain,' args.use_Adapter: ',args.use_Adapter,'args.Event_flag: ',args.Event_flag)

    if args.pretrain or args.use_Adapter:
        checkpoint = torch.load(r'F:\pytorch_project\GroupAdapter\MCAN_checkpoint\seed_2024\epochs_20\10.pth')
        pre_weights = {k:v for k,v in checkpoint['model'].items() if v.numel()==model.state_dict()[k].numel()}
        missing_keys, unexpected_keys = model.load_state_dict(pre_weights,strict=False)
        print('missing_keys: ',missing_keys,'unexpected_keys: ',unexpected_keys)

    if args.pretrain:

        not_freeze_list = ['p_fc']
        not_freeze_list += ['coconnectlayer','t_fc','f_fc','s_fc']
        #freeze parameters
        for name,param in model.named_parameters():
            if name.split('.')[0] not in not_freeze_list:
                param.requires_grad = False
        # for name, param in model.named_parameters():
        #     if name.split('.')[0] in 'coconnectlayer' and name.split('.')[1] in ['3']:
        #         param.requires_grad = True


    if args.use_Adapter:
        # freeze parameters
        not_frozen_list = ['adapter{}'.format(i) for i in range(1,5)]

        for name, param in model.named_parameters():
            if name.split('.')[2] not in not_frozen_list :
                param.requires_grad = False

    print('-------------------')
    print('trainabel params: ')
    for n,p in model.named_parameters():
        if p.requires_grad:
            print('name: ',n,' shape: ',p.shape)

    #statistic of paramters
    param_dict = get_param_number(model)
    print(param_dict)
    print('-------------------')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr=args.lr)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    best_validate_acc = 0.0
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    steps_train_acc_list = []
    print('training model')

    for epoch in range(args.start_epoch,args.epochs):

        p = float(epoch) / 100
        # init_lr = checkpoint['optimizer']['param_groups'][0]['lr']
        lr = 0.0005 / (1. + 10 * p) ** 0.75
        optimizer.param_groups[0]['lr'] = lr

        cost_vector, class_cost_vector, acc_vector = train_model_one_epoch_without_event(model, optimizer,train_loader, epoch,device, scaler)
        validate_acc_vector, valid_cost_vector, valid_pred, valid_true = evaluate_model_without_event(model,valid_loader,device)

        # valid_acc = np.mean(validate_acc_vector)
        valid_acc = sum(validate_acc_vector)/len(valid_loader.dataset)


        print('Epoch [%d/%d], Train Loss: %.4f, Valid Loss: %.4f, Train_Acc: %.4f,Validate_Acc: %.4f,lr: %.4f'
              % (
                  epoch + 1, args.epochs, np.mean(cost_vector), np.mean(valid_cost_vector),
                  np.mean(acc_vector), valid_acc, optimizer.param_groups[0]['lr']))

        with open(results_file, 'a') as f:
            train_info = 'Epoch {},Train Loss: {:.4f}, Valid Loss: {:.4f}, Train_Acc: {:.4f},  Validate_Acc: {:.4f},lr: {:.4f}' \
                .format(epoch + 1, np.mean(cost_vector), np.mean(valid_cost_vector),
                        np.mean(acc_vector), valid_acc, optimizer.param_groups[0]['lr'])
            f.write(train_info + '\n')

        # tags = ["loss", "Class Loss","accuracy","valid_acc"]
        # tb_writer.add_scalar(tags[0], np.mean(cost_vector), epoch)
        # tb_writer.add_scalar(tags[1], np.mean(class_cost_vector), epoch)
        # tb_writer.add_scalar(tags[2], np.mean(acc_vector), epoch)
        # tb_writer.add_scalar(tags[3], valid_acc, epoch)

        #use for plot_picture
        train_loss_list.append(np.mean(cost_vector))
        valid_loss_list.append(np.mean(valid_cost_vector))
        train_acc_list.append(np.mean(acc_vector))
        valid_acc_list.append(valid_acc)
        # steps_train_acc_list.extend(acc_vector)

        if  valid_acc > best_validate_acc:
            best_validate_acc =  valid_acc
            param_dict['valid_acc'] = valid_acc
            # if not os.path.exists(args.save_weights_dir):
            #     os.mkdir(args.save_weights_dir)
            # if not os.path.exists(args.save_weights_dir+'/'+str(args.seed)):
            #     os.mkdir(args.save_weights_dir+'/'+str(args.seed))

            # if not os.path.exists( './'+str(args.model_name)+'_checkpoint' + '/'+'seed_'+str(args.seed)+'/'+'epochs_'+str(args.epochs)+'/'):
            #     os.makedirs('./'+str(args.model_name)+'_checkpoint'  + '/'+'seed_'+str(args.seed)+'/'+'epochs_'+str(args.epochs)+'/')
            #
            # checkpoint = {
            #     'model':model.state_dict(),
            #     'optimizer':optimizer.state_dict(),
            #     'epoch':epoch
            # }
            # if args.amp:
            #     checkpoint['scaler '] = scaler.state_dict()
            #
            # best_validate_dir = './'+str(args.model_name)+'_checkpoint'  + '/'+'seed_'+str(args.seed)+'/'+'epochs_'+str(args.epochs)+'/'+str(epoch + 1) + '.pth'
            # torch.save(checkpoint, best_validate_dir)

            # print('valid acc: ',np.mean(valid_pred==valid_true))
            cfm = ConfusionMatrix(args.class_num)
            cfm.update(valid_pred, valid_true)
            precision_list, recall_list, f1_list = cfm.summary()
            # for i in range(args.class_num):
            #     print(f'class: {i} Precision: {precision_list[i]:.4f}  Recall: {recall_list[i]:.4f} '
            #           f'f1 : {f1_list[i]:.4f}')

            macro_precision = np.mean(np.array(precision_list))
            macro_recall = np.mean(np.array(recall_list))
            macro_f1 = np.mean(np.array(f1_list))

            with open(results_file, 'a') as f:
                # test_info = f'test_accuracy: {test_accuracy:.4f}'
                # f.write('\n' + test_info + '\n')
                f.write(f'epoch: {str(epoch + 1)}'+f' valid_acc: {np.mean(valid_pred==valid_true):.4f} macro_precision: {macro_precision:.4f} macro_recall: {macro_recall:.4f} macro_f1: {macro_f1:.4f}' + '\n')
                for i in range(args.class_num):
                    f.write(f'class: {i} Precision: {precision_list[i]:.4f} '
                            f'Recall: {recall_list[i]:.4f} '
                            f'f1: {f1_list[i]}\n')

    param_results_file = './{}_result/param_count_acc_seed_{}_epochs_{}_{}.json'.format(
        args.model_name,args.seed, args.epochs,param_dict['trainable'])
    with open(param_results_file,'a',encoding='utf-8') as file:
        json.dump(param_dict,file)
        file.write('\n')
    # train_steps_acc_results_file = "./{}_result/seed_{}_epochs_{}_steps_{}_{}.txt".format(args.model_name,args.seed,args.epochs,args.epochs*len(train_loader),args.datasets)
    # with open(train_steps_acc_results_file,'a') as f:
    #     f.write(f'train_steps: {args.epochs*len(train_loader)}\n')
    #     for data in steps_train_acc_list:
    #         f.write(str(round(data,4))+' ')
    #plot picture
    plot_loss_acc_picture(train_loss_list, valid_loss_list, train_acc_list, valid_acc_list, args)
    # plot_step_acc_pic(args.epochs*len(train_loader),steps_train_acc_list,args)

    # Test the Model
    # print('testing model')
    # model = CNN_Fusion(args, W)
    # model.load_state_dict(torch.load(best_validate_dir)['model'])
    # # print(torch.load(best_validate_dir)['optimizer']['param_groups'])
    # #    print(torch.cuda.is_available())
    # if torch.cuda.is_available():
    #     model.to(device)
    # model.eval()
    # test_score = []
    # test_pred = []
    # test_true = []
    # with torch.no_grad():
    #     for i, (test_data, test_labels,test_event) in enumerate(test_loader):
    #         test_text, test_image, test_mask, test_labels,test_event = test_data[0].to(device),\
    #                 test_data[1].to(device), test_data[2].to(device), test_labels.to(device),test_event.to(device)
    #
    #         test_outputs, domain_outputs= model(test_text, test_image, test_mask)
    #         _, test_argmax = torch.max(test_outputs, 1)
    #         if i == 0:
    #             test_score = to_np(test_outputs.squeeze())
    #             test_pred = to_np(test_argmax.squeeze())
    #             test_true = to_np(test_labels.squeeze())
    #         else:
    #             test_score = np.concatenate((test_score, to_np(test_outputs)), axis=0)
    #             test_pred = np.concatenate((test_pred, to_np(test_argmax)), axis=0)
    #             test_true = np.concatenate((test_true, to_np(test_labels)), axis=0)
    #
    # test_accuracy = np.mean(test_pred == test_true)
    # cfm = ConfusionMatrix(args.class_num)
    # cfm.update(test_pred,test_true)
    # precision_list,recall_list,f1_list = cfm.summary()
    #
    # print('test_accuracy: {:.4f}'.format(test_accuracy))
    #
    # # for i in range(args.class_num):
    # #     print(f'class: {i} Precision: {precision_list[i]:.4f}  Recall: {recall_list[i]:.4f} '
    # #           f'f1 : {f1_list[i]:.4f}')
    # #
    # # with open(results_file, 'a') as f:
    # #     test_info =f'test_accuracy: {test_accuracy:.4f}'
    # #     f.write('\n'+test_info+'\n')
    # #     for i in range(args.class_num):
    # #         f.write(f'class: {i} Precision: {precision_list[i]:.4f}  '
    # #                 f'Recall: {recall_list[i]:.4f}'
    # #                 f'f1: {f1_list[i]}\n')
    #
    # macro_precision = np.mean(np.array(precision_list))
    # macro_recall = np.mean(np.array(recall_list))
    # macro_f1 = np.mean(np.array(f1_list))
    #
    # with open(results_file, 'a') as f:
    #     f.write(f'macro_precision: {macro_precision:.4f} macro_recall: {macro_recall:.4f} macro_f1: {macro_f1:.4f}'+'\n')

    # plot_loss_acc_picture([train_loss_list, valid_loss_list, train_acc_list, valid_acc_list], args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='FT_MCAN')
    parser.add_argument('--seed', type=int, default=2040)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', default='cuda:0')
    parser.add_argument('--class_num',type=int,default=2)
    parser.add_argument('--event_num', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers',type=int,default=0)
    parser.add_argument('--max_len', type=int, default=170)
    parser.add_argument('--amp',type=bool,default=False)
    parser.add_argument('--resume',default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch',type=int,default=0)

    parser.add_argument('--bert', default='./vocab/bert/bert-base-uncased')

    parser.add_argument('--emb_type', default='bert')
    parser.add_argument('--bert_emb_dim', type=int, default=768)
    parser.add_argument('--w2v_emb_dim', type=int, default=32)
    parser.add_argument('--bert_vocab_file',
                        default='./vocab/bert/bert-base-uncased')
    parser.add_argument('--w2v_vocab_file', default='./vocab/w2v/w2v.pickle')

    parser.add_argument('--pretrain',default=False,action='store_true')
    parser.add_argument('--use_Adapter',default=False,action='store_true')

    # parser.add_argument('--datasets',default='weibo2017')
    # parser.add_argument('--train_data_path', default='./datasets/weibo2017/train.pkl')
    # parser.add_argument('--valid_data_path', default='./datasets/weibo2017/validation.pkl')
    # parser.add_argument('--test_data_path', default='./datasets/weibo2017/test.pkl')

    # parser.add_argument('--datasets', default='WeChat')
    # parser.add_argument('--train_data_path', default='./datasets/WeChat/train.pkl')
    # parser.add_argument('--valid_data_path', default='./datasets/WeChat/validation.pkl')

    # parser.add_argument('--datasets', default='Twitter')
    # parser.add_argument('--train_data_path', default='./datasets/mediaeval2015/train.pkl')
    # parser.add_argument('--valid_data_path', default='./datasets/mediaeval2015/validation.pkl')

    parser.add_argument('--datasets', default='CHECKED')
    parser.add_argument('--train_data_path', default='./datasets/CHECKED/train.pkl')
    parser.add_argument('--valid_data_path', default='./datasets/CHECKED/validation.pkl')

    parser.add_argument('--Event-flag',type=bool,default=False,help='whether to use event_label data')
    # parser.add_argument('--save_weights_dir',type=str,default='./FT_EANN_checkpoint/')

    args = parser.parse_args(['--emb_type','bert','--use_Adapter'])

    main(args)
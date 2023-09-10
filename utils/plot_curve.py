import matplotlib.pyplot as plt
import os

def plot_loss_acc_picture(train_loss,valid_loss,train_acc,valid_acc,args):
    fig,ax = plt.subplots(1,2)
    ax[0].plot(range(1,args.epochs+1,1),train_loss,'r',label='train_loss')
    ax[0].plot(range(1,args.epochs+1,1),valid_loss,'g',label='valid_loss')
    ax[0].set_title(str(args.model_name)+' batch_size:'+str(args.batch_size))
    # ax[0].set_xticks(range(args.epochs),[epoch for epoch in range(1,args.epochs+1)],rotation=30)
    ax[0].set_xticks(range(1,args.epochs+1,10),rotation=30)
    ax[0].legend()
    ax[1].plot(range(1,args.epochs+1,1),train_acc,'r',label='train_acc')
    ax[1].plot(range(1,args.epochs+1,1),valid_acc,'g',label='valid_acc')
    # ax[1].set_xticks(range(args.epochs),[epoch for epoch in range(1,args.epochs+1)],rotation=30)
    ax[1].set_xticks(range(1, args.epochs, 10), rotation=30)
    ax[1].legend()

    if not os.path.exists('./Fig'):
        os.mkdir('./Fig')
    plt.savefig('./Fig/{}_{}_Accuracy.png'.format(args.model_name,args.datasets))
    plt.show()

def plot_step_acc_pic(steps:int ,acc_list:list,args):
    plt.plot(range(steps),acc_list,'b-',marker='^')
    plt.title('{} train steps-acc'.format(args.model_name))
    plt.xticks(range(0,20000,1000),rotation=45)
    plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')

    if not os.path.exists('./Fig'):
        os.mkdir('./Fig')
    plt.savefig('./Fig/{}_{}_Step_Accuracy.png'.format(args.model_name,args.datasets))
    plt.show()

def plot_params_acc(params1,acc1,params2,acc2,params3,acc3,dataset):
    # plt.plot(params1,[acc*100 for acc in acc1],'b',marker='o',linewidth=1,label='Variable fine-tune')
    plt.plot(params2,[acc*100 for acc in acc2],'r',marker='^',linewidth=1,label='Traditional Adapter')
    plt.plot(params3,[acc*100 for acc in acc3], 'g', marker='s',linewidth=1,label='GroupAdapter')
    plt.xticks([2*10**4,2*10**5,4*10**5,6*10**5,1*10**6],['$2*10^4$','$2*10^5$','$4*10^5$','$6*10^5$','$1*10^6$'])
    plt.yticks(range(65,101,5))
    plt.xlabel('Trainable Parameters')
    plt.ylabel('Accuracy(%)')
    plt.legend(loc='lower right')
    if not os.path.exists('./Fig'):
        os.mkdir('./Fig')
    plt.savefig('./Fig/GroupAdapter_vs_Adapter_{}.png'.format(dataset))
    plt.show()

def plot_epoch_loss(epochs:int ,train_loss,valid_loss,train_loss2,valid_loss2):
    plt.plot(range(1,epochs+1),train_loss,'r',marker='o',label='FT_train_loss')
    plt.plot(range(1,epochs+1), valid_loss, 'g', marker='o',label='FT_validation_loss')
    plt.plot(range(1,epochs+1),train_loss2,'b--',marker='s',label='GroupAdapter_trainging_loss')
    plt.plot(range(1,epochs+1), valid_loss2, 'c--', marker='s',label='GroupAdapter_validation_loss')
    plt.xticks(range(1,epochs+1),[epoch for epoch in range(1,epochs+1)])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if not os.path.exists('./Fig'):
        os.mkdir('./Fig')
    plt.savefig('./Fig/{}_{}_Epoch_Loss.png'.format('MCAN','CHECKED'))
    plt.show()


def plot_location(groups,bottom,top,both,full_finetune,dataset):
    plt.plot(groups,bottom,color='y',label='bottom')
    plt.plot(groups, top, color='g',label='top')
    plt.plot(groups, both, color='b',label='both')
    plt.plot(groups, full_finetune, color='r',label='full fine-tune')
    plt.title('GroupAdapter Location({})'.format(dataset))
    plt.xticks(range(0,40,5),rotation=45)
    plt.yticks([0.5,0.6,0.7,0.8,0.9,1.0])
    plt.xlabel('Groups')
    plt.ylabel('Accuracy')
    plt.legend()
    if not os.path.exists('./Fig'):
        os.mkdir('./Fig')
    plt.savefig('./Fig/GroupAdapter_Location_{}.png'.format(dataset))
    plt.show()



# if __name__=='__main__':
#     groups = [2,4,8,16,32]
#     origin_total_params = 270596491
#     vari_params = [931691,1854315,2776939,3699563]
#     vari_acc_wechat = [0.7336,0.7988,0.8158,0.8158]
#     vari_acc_twitter = [0.6683,0.8779,0.9860,0.9860]
#     adapter_params = [20512,69760,135424,529480,1054720]
#     # adapter_total_params = [270617003,270666251,270731915,271125899,271651211]
#     adapter_acc_wechat = [0.8266,0.8285,0.8191,0.8418,0.8077]
#     adapter_acc_twitter = [0.9550,0.9742,0.9815,0.9888,0.9747]
#     group_adapter_params = [267264,136192,70656,37888,21504]
#     # group_adapter_total_params = [270863755,270732683,270667147,270634379,270617995]
#     group_adapter_acc_wechat = [0.8361,0.8351,0.8281,0.8153,0.7308]
#     group_adapter_acc_twitter = [0.9906,0.9766,0.9623,0.9030,0.6683]
#
#     plot_params_acc(vari_params,vari_acc_wechat,adapter_params,adapter_acc_wechat,group_adapter_params,group_adapter_acc_wechat,'WeChat')
#     plot_params_acc(vari_params,vari_acc_twitter,adapter_params,adapter_acc_twitter,group_adapter_params,group_adapter_acc_twitter,'Twitter')
#     group_adapter_training_loss = [0.4825,0.4450,0.3756,0.3092,0.2547,0.2112,0.1890,0.1850,0.1837,0.1821]
#     group_adapter_valid_loss = [1.5682,1.3385,1.0649,0.8575,0.6727,0.5228, 0.4484,0.4363,0.4382,0.4393]
#     FT_training_loss = [0.2135,0.1672,0.1483,0.1401,0.1270,0.0893,0.1014,0.0944,0.0771,0.0763]
#     FT_valid_loss = [0.3697, 0.4212,0.4311,0.4605,0.4568,0.5219,0.5335,0.5496,0.6764,0.7671]
#     plot_epoch_loss(10,FT_training_loss,FT_valid_loss,group_adapter_training_loss,group_adapter_valid_loss)


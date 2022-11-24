import torch
import copy
def load_model(model1,model2):
    model1=model1.state_dict()
    model2=model2.state_dict()
    model_final = copy.deepcopy(model1)
    for ((key1, value1), (key2, value2)) in zip(model1.items(), model2.items()):
        model_final[key1] = (value1 + value2) / 2
        # print(key2,value2)


    # checkpoint1 = './saved_models/tgn-attn-wikipedia-randomFalse-sample4-1.pth'
    # checkpoint2 = './saved_models/tgn-attn-wikipedia-randomFalse-sample4-2.pth'
    # checkpoint3 = './saved_models/tgn-attn-wikipedia-randomFalse-sample4-3.pth'
    # checkpoint4 = './saved_models/tgn-attn-wikipedia-randomFalse-sample4-4.pth'
    #
    # model1 = torch.load(checkpoint1,map_location=torch.device('cpu'))
    # model2 = torch.load(checkpoint2,map_location=torch.device('cpu'))
    # model3 = torch.load(checkpoint3,map_location=torch.device('cpu'))
    # model4 = torch.load(checkpoint4,map_location=torch.device('cpu'))
    #
    # model_final = model1.copy()
    # for ((key1, value1),(key2,value2),(key3,value3),(key4,value4)) in zip(model1.items(),model2.items(),model3.items(),model4.items()):
    #     # print(key1, value1)
    #     # print(key2,value2)
    #     model_final[key1] = (value1+value2+value3+value4)/4
    #     # print(key2,value2)

    # model_final=torch.load('./saved_models/tgn-attn-wikipedia-randomFalse-sample8-0.pth', map_location=torch.device('cpu'))
    # for i in range(1,8):
    #     checkpoint = './saved_models/tgn-attn-wikipedia-randomFalse-sample8-{}.pth'.format(i)
    #     cur_model=torch.load(checkpoint, map_location=torch.device('cpu'))
    #     for key, value in cur_model.items():
    #         model_final[key] += value
    #
    # for key, value in model_final.items():
    #     model_final[key] = model_final[key]/8

    return model_final
from BMW_FL import *
import pickle
import matplotlib.pyplot as plt
            
if __name__=="__main__":
    random.seed(22)
    num_per_type=[10 for i in range(10)]
    budget_each=[24,26,28,30,32]
    budget_all=140
    req_size=len(budget_each)
    train_size=1000
    test_size=2000
    val_size=2000
    batch_size=100
    input=28*28
    hidden=100
    output=10
    epochs=1
    learning_rate=0.005
    data=load_MNIST_data(num_individuals=sum(num_per_type),training_set_size=train_size,test_set_size=test_size,validation_set_size=val_size)
    workers=[]
    requester=[]
    mode=['Conflict_Detect_seperate','Conflict_Detect_overall','RAFL_seperate','RAFL_overall','BMW_FL_s','BMW_FL_g']
    
    for i in range(sum(num_per_type)):
        accuracy=random.randint(1,10)/10
        # accuracy=0.1+min(0.9,3*(int)(i/(sum(num_per_type)/6))/10)
        # print(accuracy)
        high=3+(int)(accuracy*10/3)
        low=1+(int)(accuracy*10/3)
        workers.append(Worker(input_size=input,hidden_size=hidden,output_size=output,accuracy=accuracy,data=data[i],ID=i,type_ID=get_type(num_per_type=num_per_type,ID=i),\
            range_of_bid={"high":high,"low":low},batch_size=batch_size,num_requesters=req_size))
    for i in range(req_size):
        requester.append(Requester(ID=i,budget=budget_each[i],workers=workers,num_per_type=num_per_type,num_requester=req_size,data=data[i],batch_size=batch_size))   
    req_set=Request_Set(workers=workers,requesters=requester,num_per_type=num_per_type,budget=budget_all)
    for i in range(10):
        req_set.run(mode='get_rep',size_of_selection=(int)(sum(num_per_type)/10))
    req_set.reset_for_ALG() 
    for mod in mode:
        print(mod,end='\n\n') 
        for i in range(100):
            print(f'round{i}')
            req_set.run(mode=mod)
            req_set.reset_for_ALG() 
    #get avg accuracy for all rounds 
    ac=req_set.accuracy
    # # Accuracy of overall budget ALG
    for k,v in ac.items():
        if len(v):
            if k in ['BMW_FL_g','Conflict_Detect_overall','RAFL_overall']:
                plt.plot(range(1,len(v)+1),v,label=k)
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of overall budget ALG')
    plt.legend()
    plt.savefig('Overall_budget_accuracy_1.png')
    plt.clf()
    for k,v in ac.items():  
        if len(v):
            if not(k in ['BMW_FL_g','Conflict_Detect_overall','RAFL_overall','get_rep']):
                plt.plot(range(1,len(v)+1),v,label=k)
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of seperate budget ALG')
    plt.legend()
    plt.savefig('Seperate_budget_accuracy_1.png')
    with open('model.pk','wb') as f:
        pickle.dump(ac,f)
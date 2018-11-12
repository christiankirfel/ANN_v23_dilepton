import sys

import matplotlib
print('Importing matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np
print('Importing numpy')
from copy import deepcopy
print('Importing deepcopy')
from sklearn.cross_validation import train_test_split
print('Importing train_test_split')
#Set up the name for the output files
file_extension = sys.argv[1] + '_' + str(int(sys.argv[2]) +1) + '_' + sys.argv[3]

output_path = './figures_' + file_extension + '/'


from root_numpy import root2array, tree2array
import ROOT
from numpy import *
print('Importing root')

from sklearn.preprocessing import StandardScaler
import pickle

from os import environ
import os
environ['KERAS_BACKEND'] = 'theano'

environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD


#Create the output file

if not os.path.exists(output_path):
    os.makedirs(output_path)



#Choose region

region = "2j2b"
#region = "tZ"

with open('Variables_'+sys.argv[1]+'.txt','r') as varfile:
    variableList = varfile.read().splitlines() 

def ReadOptions(region):
    with open('KerasOptions_'+region+'.txt','r') as infile:
        optionsList = infile.read().splitlines()
    OptionDict = dict()
    for options in optionsList:
		# definition of a comment
        if options.startswith('#'): continue
        templist = options.split(' ')
        if len(templist) == 2:
            OptionDict[templist[0]] = templist[1]
        else:
            OptionDict[templist[0]] = templist[1:]
    return OptionDict
    # a dictionary of options is returned

Options = ReadOptions(region)

print((variableList, Options['EventWeight']))
print (Options)



# # 2. Load data

# ## 2.1 Read ROOT file
# Two ways to read data:
# https://indico.fnal.gov/event/13497/session/1/material/slides/0?contribId=47 page 12





#Get TTree from pyROOT then convert to numpy array
#file = ROOT.TFile('data/'+str(Options['File'])+'_'+region+'_nominal.root')
#file = ROOT.TFile('lustre/lustre/'+str(Options['File'])+'_'+region+'_nominal.root')
#single lepton
file = ROOT.TFile('test_ANNinput.root')
#file = ROOT.TFile('reprocess_3j1b_nominal_kirfel_testedit.root')
#file = ROOT.TFile('MC_training/reprocessNB_3j1b_data_nominal.root')
#filelist = [ ROOT.TFile('~/internship/MC_training/' + file) for file in os.listdir('MC_training') ]
#file.ls()


# In[ ]:


''' *array* is [eventVariables, EventWeight]; *event* is [eventVariables]; *weight* is [EventWeight]'''
tree__signal = []
tree__background = []
event__signal = []
event__background = []
weight__signal = []
weight__background = []
array__signal = []
array__background = []
name__background = []

name__signal = Options['SignalTree'] #wt_DR_nominal wt_DS
name__background = Options['BackgroundTree'] #tt_nominal tt_radHi


#print name__signal
#print name__background
event_weight_branch = Options['EventWeight']
#print event_weight_branch

  
for name in name__signal:
    if file.Get(name) != None:
       tree__signal.append(file.Get(name))
       event__signal.append(tree2array(tree__signal[-1], branches=variableList, selection='1'))
       weight__signal.append(tree2array(tree__signal[-1], branches=[ Options['EventWeight'] ], selection='1'))
#     weight__signal.append(tree2array(tree__signal[-1], branches="EventWeight", selection='1'))
       array__signal.append([list(elem) for elem in zip(event__signal[-1], weight__signal[-1])])
if isinstance(name__background, list):
    for name in name__background:
        if file.Get(name) != None:
            #print 'name', name 
            tree__background.append(file.Get(name))
            event__background.append(tree2array(tree__background[-1], branches=variableList, selection='1'))
            weight__background.append(tree2array(tree__background[-1], branches=[ Options['EventWeight'] ], selection='1'))
            array__background.append([list(elem) for elem in zip(event__background[-1], weight__background[-1])])
else:
    if file.Get(name__background) != None:
       #print 'name', name 
       tree__background.append(file.Get(name__background))
       event__background.append(tree2array(tree__background[-1], branches=variableList, selection='1'))
       weight__background.append(tree2array(tree__background[-1], branches=[ Options['EventWeight'] ], selection='1'))
       array__background.append([list(elem) for elem in zip(event__background[-1], weight__background[-1])])


if bool(int(Options['UseWeight'])) is False:
    for weight in weight__signal:
        weight[:] = 1
    for weight in weight__background:
        weight[:] = 1
    print ('EventWeight set to 1')
    


# ## 2.2 Split into training and test sets

# Construct **train\_\_sample\_nominal**, **test\_\_sample\_nominal** 
# and their coresponding score **targettrain\_\_sample\_nominal**
# **targettest\_\_sample\_nominal**



'''using Options['TrainFraction'] to control fractions of training and test samples'''

def weight_ratio(weight__signal, weight__background):
    total_weight__signal = total_weight__background = 0
    for weight in weight__signal:
        print('The weight is')
        print(weight)
        total_weight__signal += sum(j[0] for j in [list(i) for i in weight])
    for weight in weight__background:
        print('The background weight is')
        print(weight)
        total_weight__background += sum(j[0] for j in [list(i) for i in weight])
    print(total_weight__signal)
    print(total_weight__background)
    return total_weight__signal / total_weight__background

ratiotWtt = weight_ratio(weight__signal, weight__background)
# ratiotWtt = sum(j[0] for j in [list(i) for i in weight__signal[0]]) / sum(j[0] for j in [list(i) for i in weight__background[0]])

train_array__signal = []
test_array__signal = []
train_array__background = []
test_array__background = []

''' *array* is [eventVariables, EventWeight]; *event* is [eventVariables]; *weight* is [EventWeight] '''
''' Construct train and test for wt_DR, tt, wt_DS '''
for array in array__signal:
    train_array, test_array = train_test_split(array, train_size=float(Options['TrainFraction']), test_size=1-float(Options['TrainFraction']), random_state = 1)
    train_array__signal.append(deepcopy(train_array))
    test_array__signal.append(deepcopy(test_array))

for array in array__background:
    train_array, test_array = train_test_split(array, train_size=float(Options['TrainFraction']), test_size=1-float(Options['TrainFraction']), random_state = 1)
    train_array__background.append(deepcopy(train_array))
    test_array__background.append(deepcopy(test_array))


train_event__signal = []
train_weight__signal = []
train_event__background = []
train_weight__background = []
test_event__signal = []
test_weight__signal = []
test_event__background = []
test_weight__background = []

for train_array in train_array__signal:
    train_event__signal.append([list(i[0]) for i in train_array])
    train_weight__signal.append([j[0]/ratiotWtt for j in [list(i[1]) for i in train_array]])

for train_array in train_array__background:
    train_event__background.append([list(i[0]) for i in train_array])
    train_weight__background.append([j[0] for j in [list(i[1]) for i in train_array]])

for test_array in test_array__signal:
    test_event__signal.append([list(i[0]) for i in test_array])
    test_weight__signal.append([j[0]/ratiotWtt for j in [list(i[1]) for i in test_array]])

for test_array in test_array__background:
    test_event__background.append([list(i[0]) for i in test_array])
    test_weight__background.append([j[0] for j in [list(i[1]) for i in test_array]])


''' Construct target for train and test for wt_DR, tt, wt_DS
    wt = 1; tt = 0 '''
train_target__signal = []
test_target__signal = []
train_target__background = []
test_target__background = []

for train_array in train_array__signal:
    train_target__signal.append(np.arange(len(train_array)))
    train_target__signal[-1][:] = 1
for test_array in test_array__signal:
    test_target__signal.append(np.arange(len(test_array)))
    test_target__signal[-1][:] = 1
for train_array in train_array__background:
    train_target__background.append(np.arange(len(train_array)))
    train_target__background[-1][:] = 0
for test_array in test_array__background:
    test_target__background.append(np.arange(len(test_array)))
    test_target__background[-1][:] = 0


''' Construct systematics for train and test for wt_DR, tt, wt_DS
    wt_DR = tt = 0; wt_DS = 1 '''
train_systematics__signal = []
test_systematics__signal = []
train_systematics__background = []
test_systematics__background = []

for train_array in train_array__signal:
    train_systematics__signal.append(np.arange(len(train_array)))
    train_systematics__signal[-1][:] = 0 if len(train_systematics__signal)==1 else 1
for test_array in test_array__signal:
    test_systematics__signal.append(np.arange(len(test_array)))
    test_systematics__signal[-1][:] = 0 if len(test_systematics__signal)==1 else 1
for train_array in train_array__background:
    train_systematics__background.append(np.arange(len(train_array)))
    train_systematics__background[-1][:] = 0 if len(train_systematics__background)==1 else 1
for test_array in test_array__background:
    test_systematics__background.append(np.arange(len(test_array)))
    test_systematics__background[-1][:] = 0 if len(test_systematics__background)==1 else 1


for i in range(len(array__signal)):
    assert (len(train_array__signal[i])+len(test_array__signal[i]) == len(array__signal[i]))
for i in range(len(array__background)):
    assert (len(train_array__background[i])+len(test_array__background[i]) == len(array__background[i]))

#Commentating out
print(('Training sample wt_DR_nominal: ', len(train_event__signal[0]), '\n',
       '                tt_nominal:   ', len(train_event__background[0])))
for i in range(1, len(train_event__signal)):
    print(('                 wt syst', i, ':   ', len(train_event__signal[i])))
for i in range(1, len(train_event__background)):
    print(('                 tt syst', i, ':   ', len(train_event__background[i])))
print(('              total nominal:   ', len(train_event__signal[0]) + len(train_event__background[0])))
print(('Test sample wt_DR_nominal: ', len(test_event__signal[0]), '\n',
       '           tt_nominal:    ', len(test_event__background[0])))
for i in range(1, len(test_event__signal)):
    print(('            wt syst', i, ':    ', len(test_event__signal[i])))
for i in range(1, len(test_event__background)):
    print(('            tt_syst', i, ':    ', len(test_event__background[i])))
print(('         total nominal:    ', len(test_event__signal[0]) + len(test_event__background[0])))



''' Construct sample, EventWeight, target, systematics of train and test
    mixing parts of wt_DR, tt, wt_DS '''

train_event__list = []
for train_event in train_event__signal:
    train_event__list.append(train_event)
for train_event in train_event__background:
    train_event__list.append(train_event)
train_event = np.vstack(train_event__list)

test_event__list = []
for test_event in test_event__signal:
    test_event__list.append(test_event)
for test_event in test_event__background:
    test_event__list.append(test_event)
test_event = np.vstack(test_event__list)

train_weight__list = []
for train_weight in train_weight__signal:
    train_weight__list.append(train_weight)
for train_weight in train_weight__background:
    train_weight__list.append(train_weight)
train_weight = np.concatenate(train_weight__list)

test_weight__list = []
for test_weight in test_weight__signal:
    test_weight__list.append(test_weight)
for test_weight in test_weight__background:
    test_weight__list.append(test_weight)
test_weight = np.concatenate(test_weight__list)

train_target__list = []
for train_target in train_target__signal:
    train_target__list.append(train_target)
for train_target in train_target__background:
    train_target__list.append(train_target)
train_target = np.concatenate(train_target__list)

test_target__list = []
for test_target in test_target__signal:
    test_target__list.append(test_target)
for test_target in test_target__background:
    test_target__list.append(test_target)
test_target = np.concatenate(test_target__list)


train_systematics__list = []
for train_systematics in train_systematics__signal:
    train_systematics__list.append(train_systematics)
for train_systematics in train_systematics__background:
    train_systematics__list.append(train_systematics)
train_systematics = np.concatenate(train_systematics__list)

test_systematics__list = []
for test_systematics in test_systematics__signal:
    test_systematics__list.append(test_systematics)
for test_systematics in test_systematics__background:
    test_systematics__list.append(test_systematics)
test_systematics = np.concatenate(test_systematics__list)


''' Data conversion of sample '''

scaler = StandardScaler()
train_event_transfered = scaler.fit_transform(train_event)
print(type(scaler))

outfolder = 'results/' + Options['Output'] + '/'
#store the content
print(outfolder + Options['Pkl'] + '.pkl')
with open(Options['Pkl'] + '.pkl', 'wb') as handle:
    pickle.dump(scaler, handle)
#load the content
#scaler = pickle.load(outfolder + open(Options['Pkl']+'.pkl', 'rb' ) )
scaler = pickle.load(open(Options['Pkl']+'.pkl', 'rb' ) )
test_event_transfered = scaler.transform(test_event)

assert (train_event_transfered.shape[1] == len(variableList))


 # 3. Simple networks




layercount = 0

simple_inputs = Input(shape=(train_event_transfered.shape[1],), name='Net_input')
simple_Dx = Dense(int(sys.argv[3]), activation="elu", name='Net_layer1')(simple_inputs)
for layercount in range(int(sys.argv[2])):
	simple_Dx = Dense(int(sys.argv[3]), activation="elu", name='Net_layer_%d' % int(layercount))(simple_Dx)
simple_Dx = Dense(1, activation="sigmoid", name='Net_output')(simple_Dx)
simple_D = Model(inputs=[simple_inputs], outputs=[simple_Dx], name='Net_model')
simple_D.compile(loss="binary_crossentropy", optimizer=SGD())
simple_D.summary()



# ## 3.2 Train



''' Train on train_event with target train_target, using train_weight as EventWeight '''
#simple_D.fit(train_event_transfered, train_target, sample_weight=train_weight, epochs=int(Options['SimpleTrainEpochs']))
#Adding Rui's loss curve plot
from IPython import display
def plot_simple_losses(losses1, losses2):
    display.clear_output(wait=True)
    display.display(plt.gcf())

    values1 = np.array(losses1)
    plt.plot(list(range(len(values1))), values1, label=r"$loss$ test", color="blue", linestyle='dashed')
    values2 = np.array(losses2)
    plt.plot(list(range(len(values2))), values2, label=r"$loss$ train", color="blue")
    plt.legend(loc="upper right")
    plt.grid()
    plt.gcf().savefig(output_path + 'simple_NN_losses' + file_extension + '.png')
    plt.gcf().clear()
    plt.show()

    
lossessimpletest = []
lossessimpletrain = []
for i in range(int(Options['SimpleTrainEpochs'])):
    simple_D.fit(train_event_transfered, train_target, sample_weight=train_weight, epochs=1)
    lossessimpletest.append(simple_D.evaluate(test_event_transfered, test_target, sample_weight=test_weight, verbose=0))
    lossessimpletrain.append(simple_D.evaluate(train_event_transfered, train_target, sample_weight=train_weight, verbose=0))
    
    plot_simple_losses(lossessimpletest, lossessimpletrain)
# ## 3.3 Test



''' Apply training results to test sample; and training sample for checking '''
from sklearn.metrics import roc_auc_score
predicttest__simple_D = simple_D.predict(test_event_transfered)
predicttrain__simple_D = simple_D.predict(train_event_transfered)


# ## 3.4 Calculate and plot ROC



from sklearn.metrics import roc_curve, auc

print(('Traing ROC: ', roc_auc_score(train_target, predicttrain__simple_D)))
print(('Test ROC:   ', roc_auc_score(test_target, predicttest__simple_D)))




''' Plot ROC '''

train__false_positive_rate, train__true_positive_rate, train__thresholds = roc_curve(train_target, predicttrain__simple_D)
test__false_positive_rate, test__true_positive_rate, test__thresholds = roc_curve(test_target, predicttest__simple_D)
train__roc_auc = auc(train__false_positive_rate, train__true_positive_rate)
test__roc_auc = auc(test__false_positive_rate, test__true_positive_rate)


plt.title('Receiver Operating Characteristic')
plt.plot(train__false_positive_rate, train__true_positive_rate, 'g--', label='Train AUC = %0.2f'% train__roc_auc)
plt.plot(test__false_positive_rate, test__true_positive_rate, 'b', label='Test AUC = %0.2f'% test__roc_auc)

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.,1.])
plt.ylim([-0.,1.])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()
plt.gcf().savefig(output_path + 'simple_ROC_' + file_extension + '.png')
plt.gcf().clear()



# ## 3.5 Plot traing and test distributions



''' Plot NN output '''
xlo, xhi, bins = float(Options['xlo']), float(Options['xhi']), int(Options['bins'])
SignalName, BckgrdName = Options['SignalName'], Options['BackgroundName']

plt.subplot(1, 2, 1)
plt.hist(predicttrain__simple_D[train_target == 1], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=SignalName+' training')
plt.hist(predicttrain__simple_D[train_target == 0], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' training')
plt.hist(predicttest__simple_D[test_target == 1],   range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=SignalName+' test', linestyle='dashed')
plt.hist(predicttest__simple_D[test_target == 0],   range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' test', linestyle='dashed')
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('Simple_D response', horizontalalignment='left', fontsize='large')
plt.title('Absolute')

plt.subplot(1, 2, 2)
plt.hist(predicttrain__simple_D[train_target == 1], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=SignalName+' training')
plt.hist(predicttrain__simple_D[train_target == 0], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' training')
plt.hist(predicttest__simple_D[test_target == 1],   range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=SignalName+' test', linestyle='dashed')
plt.hist(predicttest__simple_D[test_target == 0],   range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' test', linestyle='dashed')
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('Simple_D response', horizontalalignment='left', fontsize='large')
plt.title('Normalised')
plt.gcf().savefig(output_path + 'simple_NN_' + file_extension + '.png')
plt.gcf().clear()
plt.gcf().clear()


print(('Test sample wt_DR_nominal: ', len(predicttest__simple_D[test_target == 1]), '\n'))
      #'           tt_nominal:    ', len(predicttest__simple_D[test_target == 0]), '\n',
      #'           total:         ', len(predicttest__simple_D))





plt.subplot(1, 2, 1)
plt.hist(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 0)], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=SignalName+' norm')
plt.hist(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 0)], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' norm')
plt.hist(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 1)], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=SignalName+' syst', linestyle='dashed')
plt.hist(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 1)], range=[xlo,xhi], bins=bins, histtype="step", normed=0, label=BckgrdName+' syst', linestyle='dashed')
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend() 

plt.xlabel('Simple_D response', horizontalalignment='left', fontsize='large')
plt.title('Absolute')

plt.subplot(1, 2, 2)
plt.hist(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 0)], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=SignalName+' norm')
plt.hist(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 0)], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' norm')
plt.hist(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 1)], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=SignalName+' syst', linestyle='dashed')
plt.hist(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 1)], range=[xlo,xhi], bins=bins, histtype="step", normed=1, label=BckgrdName+' syst', linestyle='dashed')
plt.ylim(0, plt.gca().get_ylim()[1] * float(Options['yScale']))
plt.legend()
plt.xlabel('Simple_D response', horizontalalignment='left', fontsize='large')
plt.title('Normalised')
#plt.show()
plt.gcf().savefig(output_path + 'simple_sys_' + file_extension + '.png')
plt.gcf().clear()


print(('Test sample wt_DR_nominal: ', len(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 0)]), '\n'))
      #'            wt_DS_nominal: ', len(predicttest__simple_D[logical_and(test_target == 1, test_systematics == 1)]), '\n',
      #'               tt_nominal: ', len(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 0)]), '\n',
      #'           tt systematics: ', len(predicttest__simple_D[logical_and(test_target == 0, test_systematics == 1)])
     



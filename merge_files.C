#include <iostream>
#include <algorithm>
#include <fstream>
#include <mutex>
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
using namespace std;


//This is a script to open root files and merge them into a new file
//There should be a function for cuts
//There should be the possibility to add new variables
//For now my old script can do it

//have a list of dsids, read from that
//do that for nominal and sys


void merge_files() {

cout << "Der Gott, der Eisen wachsen liess" << endl;

TChain * chain_wt_nominal;
TChain * chain_tt_nominal; 
TChain * chain_wt_DS; 
TString output_test;
output_test="test_merge.root";

TFile* output_file = new TFile(output_test,"recreate");

cout << "Der wollte keine Knechte" << endl;

chain_wt_nominal = new TChain("WtLoop_nominal");
    chain_wt_nominal->Add("410646.full.a.root");
    chain_wt_nominal->Add("410646.full.d.root");
    chain_wt_nominal->Add("410647.full.a.root");
    chain_wt_nominal->Add("410647.full.d.root");

cout << "Drum gab er Saebel Schwert und Spiess" << endl;

chain_tt_nominal = new TChain("WtLoop_nominal");
    chain_tt_nominal->Add("410472.full.a.root");
    chain_tt_nominal->Add("410472.full.d.root"); 

cout << "Dem Mann in seine Rechte" << endl;

chain_wt_DS = new TChain("WtLoop_nominal");
    chain_wt_DS->Add("410654.full.a.root");
    chain_wt_DS->Add("410654.full.d.root");
    chain_wt_DS->Add("410655.full.a.root");
    chain_wt_DS->Add("410655.full.d.root");

cout << "Drum gab er ihm den frohen Mut" <<endl;

/*
chain2 = new TChain("WtLoop_JET_CategoryReduction_JET_BJES_Response__1down");
    chain2->Add("410646.fast.a.root");
    chain2->Add("410646.fast.d.root");
    chain2->Add("410470.fast.a.root");
    chain2->Add("410470.fast.d.root");


chain3 = new TChain("WtLoop_JET_CategoryReduction_JET_BJES_Response__1up");
    chain3->Add("410646.fast.a.root");
    chain3->Add("410646.fast.d.root");
    chain3->Add("410470.fast.a.root");
    chain3->Add("410470.fast.d.root");
*/


output_file->cd();
TTree *test_tree1= chain_wt_nominal->CloneTree();
delete chain_wt_nominal;
test_tree1 ->SetName("wt_nominal");
TTree *test_tree2= chain_tt_nominal->CloneTree();
delete chain_tt_nominal;
test_tree2 ->SetName("tt_nominal");
TTree *test_tree3= chain_wt_DS->CloneTree();
test_tree3 ->SetName("wt_DS");

cout << "Kirmeszelt" << endl;

output_file->Write();
output_file->Close();



//vector<string> DSID_list = {"410464"};

//for(vector<string>::const_iterator i = features.begin(); i != features.end(); ++i) {
//    cout << "Current DSID is" << *i << "!";

//need to make output variable
//output_test= *i
//}


}
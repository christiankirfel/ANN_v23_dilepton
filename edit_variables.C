void edit_variables()
{	//List of trees you want to keep
	//TString trname[132]={"wt_DR_nominal","wt_DS","tt_nominal","wt_JET_21NP_JET_EffectiveNP_1__1down","wt_JET_21NP_JET_EffectiveNP_1__1up"};
	//TString trname[132]={"wt_DR_nominal","wt_DS","tt_nominal","tt_JET_21NP_JET_EffectiveNP_1__1down","tt_JET_21NP_JET_EffectiveNP_1__1up"};
	TString trname[132]={"tt_nominal","wt_nominal","wt_DS"};
	
	//TFile *f_all=new TFile("reprocess_3j1b_v15_20180123.root");//input
	//TFile *f_sub=new TFile("reprocess_3j1b_nominal_kirfel_testedit.root","RECREATE");//output
	
	TFile *f_all=new TFile("test_merge.root");//input
	TFile *f_sub=new TFile("test_ANNinput.root","RECREATE");//output
	
	//input variables
	float Abs_Eta_Jet1 = 0;
	float Abs_Eta_Jet2 = 0;
	float Abs_Eta_Jet3 = 0;
	float Abs_Eta_Lep1 = 0;
	float px_Jet1 = 0;
	float px_Jet2 = 0;
	float px_Jet3 = 0;
	float px_Lep1 = 0;
	float py_Jet1 = 0;
	float py_Jet2 = 0;
	float py_Jet3 = 0;
	float py_Lep1 = 0;
	
	//Number cartesians
	
	int quadrant_1 = 1;
	int quadrant_2 = 1;
	int quadrant_3 = 1;
	int quadrant_Lep =1;
	
	
	float eta_Jet1_address = 0;
	float eta_Jet2_address = 0;
	float eta_Jet3_address = 0;
	float eta_Lep1_address = 0;
	float pT_Jet1_address = 0;
	float pT_Jet2_address = 0;
	float pT_Jet3_address = 0;
	float pT_Lep1_address = 0;
	float phi_Jet1_address = 0;
	float phi_Jet2_address = 0;
	float phi_Jet3_address = 0;
	float phi_Lep1_address = 0;
	
	for(int i=0;i<3;i++)
	{

		TTree *tr_all=(TTree*)f_all->Get(trname[i]);
		TTree *tr_sub;//=new TTree("infmom");

		tr_sub=tr_all->CopyTree("reg2j2b");
		//tr_sub=tr_all->CopyTree("pass_3j1b&&pass_TruthMatch");
		//CloneTree(0)
		
		TBranch *bnew1 = tr_sub->Branch("absEta_Jet1", &Abs_Eta_Jet1, "absEta_Jet1/F");
		TBranch *bnew2 = tr_sub->Branch("absEta_Jet2", &Abs_Eta_Jet2, "absEta_Jet2/F");
		//TBranch *bnew3 = tr_sub->Branch("absEta_Jet3", &Abs_Eta_Jet3, "absEta_Jet3/F");
		TBranch *bnew4 = tr_sub->Branch("absEta_Lep1", &Abs_Eta_Lep1, "absEta_Lep1/F");
		TBranch *bnew5 = tr_sub->Branch("px_Jet1", &px_Jet1, "px_Jet1/F");
		TBranch *bnew6 = tr_sub->Branch("px_Jet2", &px_Jet2, "px_Jet2/F");
		//TBranch *bnew7 = tr_sub->Branch("px_Jet3", &px_Jet3, "px_Jet3/F");
		TBranch *bnew8 = tr_sub->Branch("px_Lep1", &px_Lep1, "px_Lep1/F");
		TBranch *bnew9 = tr_sub->Branch("py_Jet1", &py_Jet1, "py_Jet1/F");
		TBranch *bnew10 = tr_sub->Branch("py_Jet2", &py_Jet2, "py_Jet2/F");
		//TBranch *bnew11 = tr_sub->Branch("py_Jet3", &py_Jet3, "py_Jet3/F");
		TBranch *bnew12 = tr_sub->Branch("py_Lep1", &py_Lep1, "py_Lep1/F");
		
		tr_sub->SetBranchStatus("*",0);
		
		
		//new branches
		tr_sub->SetBranchStatus("absEta_Jet1",1);
		tr_sub->SetBranchStatus("absEta_Jet2",1);
		//tr_sub->SetBranchStatus("absEta_Jet3",1);
		tr_sub->SetBranchStatus("absEta_Lep1",1);
		tr_sub->SetBranchStatus("px_Jet1",1);
		tr_sub->SetBranchStatus("px_Jet2",1);
		//tr_sub->SetBranchStatus("px_Jet3",1);
		tr_sub->SetBranchStatus("px_Lep1",1);
		tr_sub->SetBranchStatus("py_Jet1",1);
		tr_sub->SetBranchStatus("py_Jet2",1);
		//tr_sub->SetBranchStatus("py_Jet3",1);
		tr_sub->SetBranchStatus("py_Lep1",1);
		
		//old branches needed for computation
		tr_sub->SetBranchStatus("eta_jet1",1);
		tr_sub->SetBranchStatus("eta_jet2",1);
		//tr_sub->SetBranchStatus("eta_jet3",1);
		tr_sub->SetBranchStatus("eta_lep1",1);
		tr_sub->SetBranchStatus("pT_jet1",1);
		tr_sub->SetBranchStatus("pT_jet2",1);
		//tr_sub->SetBranchStatus("pT_jet3",1);
		tr_sub->SetBranchStatus("pT_lep1",1);
		tr_sub->SetBranchStatus("phi_jet1",1);
		tr_sub->SetBranchStatus("phi_jet2",1);
		//tr_sub->SetBranchStatus("phi_jet3",1);
		tr_sub->SetBranchStatus("phi_lep1",1);
		
		
		//test to access variables
		tr_sub->SetBranchAddress("eta_jet1", &eta_Jet1_address);
		tr_sub->SetBranchAddress("eta_jet2", &eta_Jet2_address);
		//tr_sub->SetBranchAddress("eta_jet3", &eta_Jet3_address);
		tr_sub->SetBranchAddress("eta_lep1", &eta_Lep1_address);
		tr_sub->SetBranchAddress("pT_jet1", &pT_Jet1_address);
		tr_sub->SetBranchAddress("pT_jet2", &pT_Jet2_address);
		//tr_sub->SetBranchAddress("pt_jet3", &pT_Jet3_address);
		tr_sub->SetBranchAddress("pT_lep1", &pT_Lep1_address);
		tr_sub->SetBranchAddress("phi_jet1", &phi_Jet1_address);
		tr_sub->SetBranchAddress("phi_jet2", &phi_Jet2_address);
		//tr_sub->SetBranchAddress("phi_jet3", &phi_Jet3_address);
		tr_sub->SetBranchAddress("phi_lep1", &phi_Lep1_address);
		
		Int_t nentries = tr_sub->GetEntries();
		for (Int_t event_counter=0;event_counter<nentries;event_counter++) {
		tr_sub->GetEvent(event_counter);
		
		int quadrant_1 = 1;
		int quadrant_2 = 1;
		int quadrant_3 = 1;
		int quadrant_Lep =1;
		
		
		//Compute new variables
		Abs_Eta_Jet1 = fabs( eta_Jet1_address );
		Abs_Eta_Jet2 = fabs( eta_Jet2_address );
		//Abs_Eta_Jet3 = fabs( eta_Jet3_address );
		Abs_Eta_Lep1 = fabs( eta_Lep1_address );
		
		if( phi_Jet1_address > (M_PI/2)){ phi_Jet1_address = M_PI - phi_Jet1_address;
										quadrant_1 = 2;}
		if( phi_Jet1_address > M_PI  ){ phi_Jet1_address = phi_Jet1_address - M_PI;
										quadrant_1 = 3;}
		if( phi_Jet1_address > ((3*M_PI)/2)){ phi_Jet1_address = 2*M_PI - phi_Jet1_address;
											quadrant_1 = 4;}

		if( phi_Jet2_address > (M_PI/2)){ phi_Jet2_address = M_PI - phi_Jet2_address;
										quadrant_2 = 2;}
		if( phi_Jet2_address > M_PI  ){ phi_Jet2_address = phi_Jet2_address - M_PI;
										quadrant_2 = 3;}
		if( phi_Jet2_address > ((3*M_PI)/2)){ phi_Jet2_address = 2*M_PI - phi_Jet2_address;
											quadrant_2 = 4;}

		if( phi_Jet3_address > (M_PI/2)){ phi_Jet3_address = M_PI - phi_Jet3_address;
										quadrant_3 = 2;}
		if( phi_Jet3_address > M_PI  ){ phi_Jet3_address = phi_Jet3_address - M_PI;
										quadrant_3 = 3;}
		if( phi_Jet3_address > ((3*M_PI)/2)){ phi_Jet3_address = 2*M_PI - phi_Jet3_address;
											quadrant_3 = 4;}

		if( phi_Lep1_address > (M_PI/2)){ phi_Lep1_address = M_PI - phi_Lep1_address;
										quadrant_Lep = 2;}
		if( phi_Lep1_address > M_PI  ){ phi_Lep1_address = phi_Lep1_address - M_PI;
										quadrant_Lep = 3;}
		if( phi_Lep1_address > ((3*M_PI)/2)){ phi_Lep1_address = 2*M_PI - phi_Lep1_address;
											quadrant_Lep = 4;}

		px_Jet1 = pT_Jet1_address * sin( phi_Jet1_address );
		px_Jet2 = pT_Jet2_address * sin( phi_Jet2_address );
		//px_Jet3 = pT_Jet2_address * sin( phi_Jet3_address );
		px_Lep1 = pT_Lep1_address * sin( phi_Lep1_address );
		
		py_Jet1 = pT_Jet1_address * cos( phi_Jet1_address );
		py_Jet2 = pT_Jet2_address * cos( phi_Jet2_address );
		//py_Jet3 = pT_Jet3_address * cos( phi_Jet3_address );
		py_Lep1 = pT_Lep1_address * cos( phi_Lep1_address );
		
		if( quadrant_1 == 2 ) px_Jet1 = -px_Jet1;
		if( quadrant_1 == 3 ){ 
								px_Jet1 = -px_Jet1;
								py_Jet1 = -py_Jet1;
							}
		if( quadrant_1 == 4 ) py_Jet1 = -py_Jet1;
		
		
		if( quadrant_2 == 2 ) px_Jet2 = -px_Jet2;
		if( quadrant_2 == 3 ){ 
								px_Jet2 = -px_Jet2;
								py_Jet2 = -py_Jet2;
							}
		if( quadrant_2 == 4 ) py_Jet2 = -py_Jet2;
		
		
		//if( quadrant_3 == 2 ) px_Jet3 = -px_Jet3;
		//if( quadrant_3 == 3 ){ 
		//						px_Jet3 = -px_Jet3;
		//						py_Jet3 = -py_Jet3;
		//					}
		//if( quadrant_3 == 4 ) py_Jet3 = -py_Jet3;
		
		
		if( quadrant_Lep == 2 ) px_Lep1 = -px_Lep1;
		if( quadrant_Lep == 3 ){ 
								px_Lep1 = -px_Lep1;
								py_Lep1 = -py_Lep1;
							}
		if( quadrant_Lep == 4 ) py_Lep1 = -py_Lep1;
		 
		
		//Fill branches
		bnew1->Fill();
		bnew2->Fill();
		//bnew3->Fill();
		bnew4->Fill();
		bnew5->Fill();
		bnew6->Fill();
		//bnew7->Fill();
		bnew8->Fill();
		bnew9->Fill();
		bnew10->Fill();
		//bnew11->Fill();
		bnew12->Fill();
		}
		
		tr_sub->SetBranchStatus("*",1);//reactivate
		cout<<trname[i]<<endl;
		tr_sub->Write("", TObject::kOverwrite);
		delete tr_sub;
	}
	f_sub->Close();
}

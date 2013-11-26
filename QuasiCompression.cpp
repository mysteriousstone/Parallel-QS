/*
 * QuasiCompression.cpp
 *
 *  Created on: Apr 30, 2013
 *      Author: shi
 */

/*
 * This is the C++ implementation of quasi-succinct.
 * For the algorithm, please see http://arxiv.org/pdf/1206.4300.pdf
 *
 */
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <cmath>
using namespace std;

typedef unsigned int uint;

struct quasilexicon
{
	char name[20];
	int len;
	int bw;
	uint offset;
	int upperlen;
	int lowerlen;
};

struct lexi
{
	char name[20];
	int len;
};

vector<quasilexicon> qlexicon;


//ulva: calculate the values in upper bits array and lower bits array
void ulva(uint* in, uint* upper, uint* lower, uint bw, uint size){
	*upper = (*in)>>bw;
	*lower = (*in)&(uint)(pow((float)2, (float)bw)-1);
	in++;
	upper++;
	lower++;
	for(uint i=1; i<=size-1; i++){
		*upper = ((*in)>>bw) - (*(in-1)>>bw);
		*lower =  (*in)&(uint)(pow((float)2, (float)bw)-1);//cout<<"!!!"<<*lower<<" ";
		in++;
		upper++;
		lower++;

	}
}



//packing upper bits array (unary codes)
void packupper(uint* in, uint* out, uint size){
	int counter = 32;
	for (uint i = 0; i<=size-1; i++){
	if ((int)(counter - *in -1  )>=0){
		*out = (*out<<(*in+1)) | 1;
		counter= counter - (*in) -1;
		in++;
	}
	else{
		*out = *out<<counter;
		out++;
		*in -= counter;
		counter =32;
		for(uint i =0; i< (*in)/counter;i++){
			*out =0;
			out++;
		}
		(*in) %= counter;
		*out = (*out<<(*in+1))|1;
		counter = counter - (*in) -1;
		in++;
	}
  }
	*out = *out <<counter;
}



//packing lower bits array (binary codes)
void pack(uint* in, uint* out, uint bw, uint size){
	int counter = 32;
	for(uint i = 0; i <= size-1; i++ ){
		if((int)(counter - bw)>0){
			*out = (*out<<bw) | *in;
			counter -= bw;
			in++;
		}
		else{
			*out = *out<<counter;
			*out = *out | (*in>>(bw - counter));
			out++;
			*out =0;
			*in = *in & ((uint)(pow((float)2, (float)(bw - counter)) -1));
			*out = *in;
			in++;
			counter = 32 - (bw-counter);
		}
	}
	*out = *out <<counter;
}

ofstream qdata("/home/shi/newlists1/qdata.dat", ios::binary); //output data




//General compression procedure of Quasi-Succinct algorithm.

void quasi(uint doc[], int len, string name){
	//uint a[5] = {5,8,8,15,35};
	uint c = (uint)len;
	
	uint* uptr = new uint[c];
	uint* lptr = new uint[c];
	uint* ua = new uint[c];
	uint* inptr = &doc[0];
	uint* lp = &lower[0];
	uint bits = (log(doc[c-1]/c))/log(2);
	
	ulva(inptr, uptr, lptr, bits, c);
	uint lower[c*bits/32 + 1];
	pack(lptr, lp, bits, c);
	packupper(uptr, ua, c);

	int d  = c;
	for( int i = c-1; i>0; i--)
	{
		if (upperarray[i] == 0)
		{
			d -=1;
		}
		else
		{
			break;
		}
	}
	uint* upper;
	upper = new uint[d];
	for(int i = 0; i<d; i++)
	{
		upper[i] = ~upperarray[i];
	}

	quasilexicon ql;
	strcpy(ql.name, name.c_str());
	
	ql.len = (int)len;
	ql.bw = (int)bits;
	ql.lowerlen = (int)c*bits/32+1;
	ql.upperlen = (int)d;
	ql.offset = 0;

	qlexicon.push_back(ql);


	qdata.write((char*) &upper[0], d*sizeof(int));
	qdata.write((char*) &lower[0], (c*bits/32+1)*sizeof(int));

	delete[] uptr;
	delete[] lptr;
	delete[] ua;
	delete[] upper;
}



int main()
{
	lexi a;
	ifstream in("/home/shi/newlists1/lexicon.dat", ios::binary);
	ifstream data("/home/shi/newlists1/data.dat", ios::binary);
	ofstream qin("/home/shi/newlists1/qlexicon.dat", ios::binary);
	uint AllLength = 0;
	while(in.good())
	{
		in.read((char*) &a,sizeof(lexi));
		if(!in.eof())
		{
			cout<<"<<"<<a.name<<" "<<a.len<<">>"<<" ";
			string name(a.name);
			uint* doc;
			AllLength += a.len;
			doc = new uint[a.len];
			data.read((char*) & doc[0], (a.len)*sizeof(int));
						cout<<doc[0]<<" "<<doc[a.len-1]<<endl;
						quasi(doc,a.len,name );
			delete doc;
			data.seekg(sizeof(uint), ios::cur);

		}
	}
	for(int i = 1; i<(int)qlexicon.size(); i++)
	{
		qlexicon[i].offset = qlexicon[i-1].offset + qlexicon[i-1].lowerlen + qlexicon[i-1].upperlen;
		
	//To view the metadata of each compressed list, uncomment the following three lines.
	/*	cout<<"%%%%%%"<<endl;
	    cout<<qlexicon[i].name<<" "<<qlexicon[i].len<<" "<<qlexicon[i].bw<<" "<<qlexicon[i].lowerlen<<" "<<qlexicon[i].upperlen<<" "<<qlexicon[i].offset<<endl;
		cout<<"%%%%%%"<<endl;
	*/
	}
	qin.write((char*)&qlexicon[0], qlexicon.size()*sizeof(quasilexicon));
	//cout<<qlexicon.size()<<endl;
	//cout<<AllLength;


}

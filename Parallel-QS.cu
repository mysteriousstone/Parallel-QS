/*
 * This is the source code running on the GPU. To compile this source code, you will need 
 * to have a NVIDIA GPU which supports CUDA. In addition, to run code you will need to
 * configure CUDPP library (CUDA Data Parallel Primitives Library)on your machine.
 *
 *
 * The detailed introduction of the new Parallel-QS method can be found in the ".pdf" file.
 */



#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include "/home/shi/NVIDIA_CUDA-5.0_Samples/common/inc/exception.h"
#include "/home/shi/NVIDIA_CUDA-5.0_Samples/common/inc/timer.h"
#include "/home/shi/timerlib/helper_timer.h"
#include "/home/shi/cuda/cuda_by_example/common/book.h"
using namespace std;




// includes, project
#include "cudpp.h"

#include <string>

// includes, project
#include "cudpp.h"

#include <string>

using namespace std;

typedef unsigned int uint;
struct Quasilexicon
{
	char name[20];
	uint len;
	uint bw;
	uint offset;
	uint upperlen;
	uint lowerlen;
} quasi;




//unpack lower bits array
//special bit width of "0, 1, 2, 4, 8, 16"
__global__ void unpack0(uint*p, uint* w, uint num, uint* blockprefix, uint* result)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < (int)num)
	{
		*p = *w;
	//	tid += blockDim.x * gridDim.x;
		result[tid] = blockprefix[tid] + p[tid];
	}
	__syncthreads();
}

//-------------------------------------------------------------------

__global__ void unpack1(uint* p, uint* w, uint num, uint* blockprefix, uint* result)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < (int)num)//for (i=0; i<(int)num; i += 32, p += 32, w++ )
	{
		p[tid] = (w[tid/32] >> (31-tid%32))&1;
		result[tid] =( blockprefix[tid] * 2)+ p[tid];
	//	tid += blockDim.x * gridDim.x;
	}
}

//-------------------------------------------------------------------

__global__ void unpack2(uint *p, uint *w, uint num, uint* blockprefix, uint* result)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < (int)num)//for (i = 0; i < (int)num; i += 32, p += 32, w += 2)
  {
	p[tid] = (w[tid/16] >> (30-2*(tid%16))) &3;
	result[tid] =( blockprefix[tid] * 4)+ p[tid];
 //   tid += blockDim.x * gridDim.x;
  }
}

//-------------------------------------------------------------------

__global__ void unpack4(uint *p, uint *w, uint num, uint* blockprefix, uint* result)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < (int)num)// for (i = 0; i < (int)num; i += 32, p += 32, w += 4)
  {
	p[tid] = (w[tid/8] >> (28-4*(tid%8))) & 15;
	result[tid] =( blockprefix[tid] * 16)+ p[tid];
   // tid += blockDim.x * gridDim.x;
  }
	__syncthreads();
}

//-------------------------------------------------------------------

__global__ void unpack8(uint *p, uint *w, uint num, uint* blockprefix, uint* result)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < (int)num)// for (i = 0; i < (int)num; i += 32, p += 32, w += 8)
  {
		p[tid] = (w[tid/4] >> (24-8*(tid%4))) & 255;
		result[tid] =( blockprefix[tid] * 256)+ p[tid];
		//tid += blockDim.x * gridDim.x;
  }
	__syncthreads();
}

//-------------------------------------------------------------------

__global__ void unpack16(uint *p, uint *w, uint num, uint* blockprefix, uint* result)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < (int)num)// for (i = 0; i < (int)num; i += 32, p += 32, w += 16)
  {

	p[tid] = (w[tid/2] >> (16 - 16*(tid%2))) & 65535;
	result[tid] =( blockprefix[tid] * 65536)+ p[tid];
   //tid += blockDim.x * gridDim.x;
  }
	__syncthreads();
}

//-------------------------------------------------------------------
// common bit width

__global__ void unpackdiffbw(uint *p, uint*w, uint num, uint bw, uint* blockprefix, uint* result)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if ( tid < (int)num)
	{
		int mask = (1<<bw) -1 ;
		int index = (bw * tid) >> 5;
		int offset = 32 - bw - ((tid * bw) & 31);//32*(index+1) - (tid % 32) * bw - bw;
		if(offset >= 0)
			p[tid] = (w[index] >> offset) & mask;

		else
		{
			offset = -offset;
			p[tid] = ((w[index] <<  offset)|(w[index +1] >> (32 -offset))) & mask;
		}
		result[tid] =( blockprefix[tid] * (1 << bw ))+ p[tid];
		//tid += blockDim.x * gridDim.x;
	}
}

//-------------------------------------------------------------------

__global__ void GetUnary(uint* prefixsum, uint* index, uint* out, uint size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid < size)
	{
		if(index[tid] == 0)
		{
			out[tid - prefixsum[tid]] = prefixsum[tid];
		}
	//	else index[tid] = 1;
	//		(index[tid] == 0) ? out[tid - prefixsum[tid]] = prefixsum[tid]:break;
		tid += blockDim.x * gridDim.x;
		//__syncthreads();
	}
}

__global__ void Add (uint *binary, uint *unary, uint *result, uint bw, uint size)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	while ( tid < size)
	{
		result[tid] =( unary[tid] * (1 << bw ))+ binary[tid];
		tid += blockDim.x * gridDim.x;
	}
	__syncthreads();
}


__global__ void Blockprefixsum(uint* unary,  uint* prefixsum, uint* output,  uint size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size)
	{
		int prefix(0);
		int bit(0);
		int index;
		//uint bb = unary[tid];
		prefix = prefixsum[tid];
		//int blockid = 32*tid;
		for(int i=0; i<32; i++)
		{
			//bit = bb&mask[i];
			bit = (unary[tid]>>(31-i))&1;
			if(bit == 0)
			{
				index = 32*tid+i-prefix;
				output[index] = prefix;
			}
			else
			{
				prefix++;
			}
		}

	//	tid += blockDim.x * gridDim.x;
	}
	__syncthreads();
}

__global__ void readbits(uint *in, uint *out, uint length)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < 32*length)
	{
		out[tid] = (in[tid/32] >> (31 - tid%32)) &1;
		//out2[tid] =(~in[tid/32] >> (31 - tid%32)) &1;
		tid += blockDim.x * gridDim.x;

	}
}

__global__ void BitCount(uint* in, uint* out, uint size)
{
	 uint tid = threadIdx.x + blockIdx.x * blockDim.x;
     uint uCount;
     while(tid < size)
     {
     uCount = in[tid] - ((in[tid] >> 1) & 033333333333) - ((in[tid] >> 2) & 011111111111);
     out[tid] =((uCount + (uCount >> 3)) & 030707070707) % 63;
     tid += blockDim.x * gridDim.x;
     }
}

void unpack ( uint* p, uint* w, uint num, uint bw, uint* u, uint* r)
{
	switch(bw)
	{
	case 0: unpack0<<<num/512+1, 512>>>(p, w, num, u, r); break;
	case 1: unpack1<<<num/512+1, 512>>>(p, w, num, u, r); break;
	case 2: unpack2<<<num/512+1, 512>>>(p, w, num, u, r); break;
	case 3: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 4: unpack4<<<num/512+1, 512>>>(p, w, num, u, r); break;
	case 5: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 6: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 7: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 8: unpack8<<<num/512+1, 512>>>(p, w, num, u, r); break;
	case 9: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 10: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 11: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 12: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 13: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 14: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 15: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 16: unpack16<<<num/512+1, 512>>>(p, w, num, u, r); break;
	case 17: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 18: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 19: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 20: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 21: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 22: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 23: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;
	case 24: unpackdiffbw<<<num/512+1, 512>>>(p, w, num, bw, u, r); break;

	}
}


int main()
{

	StopWatchLinux timer, timer1, timer2;
	StopWatchLinux ti1, ti2, ti3, ti4, ti5, ti6, ti7;
	timer1.start();
	float total1(0), total2(0), total3(0), total4(0), total5(0), total6(0), total(0);
	float t1, t2, t3, t4, t5, t6, t7;
	float totaltime;
	float avg_t1, avg_t2, avg_t3, avg_t4, avg_t;
	float totals, avgs;
	ofstream cout;
	cout.open("/home/shi/result/512");


	//read unary data into memory
	ifstream qdata("/home/shi/newlists1/qdata.dat", ios::binary|ios::ate);
	ifstream data("/home/shi/newlists1/data.dat", ios::binary);
	int size;
	if(qdata == NULL)
		cout<<"bad";
	//if(data == NULL)
	//	cout<<"bad";
	else
	{
		size = (int) qdata.tellg();
		qdata.seekg (0, ios::beg);
		//cout<<size;
	}

	uint* buffer = new uint[size/4];
	qdata.read((char*) & buffer[0], (size/4)*sizeof(uint));



	//read metadata
	ifstream qlexicon("/home/shi/newlists1/qlexicon.dat", ios::binary);

	uint* packedUnary = new uint[25000000];
	uint* docptr = new uint[25000000];
	uint* binaryptr = new uint[25000000];
	uint* h_odata = new uint[25000000] ;
	uint* h_unaryBits = new uint[25000000];
	uint* h_unary = new uint[25000000];
	uint* h_packedUnary = new uint[25000000];
	uint* h_docptr  = new uint[25000000];
	uint* h_result = new uint[25000000];
	uint* h_ZeroIndex = new uint[25000000];
	uint* h_odata1 = new uint[25000000];
	uint* h_odata2 = new uint[25000000];
	uint* mask = new uint[32];
	for(int i = 0; i<32; i++)
	{
		mask[i] = 1<<(31-i);
	}


	uint* dev_doc;
	uint* dev_binary;
	uint* dev_unary;// device side packed unary bits array
	uint* dev_unaryBits; // quasi.upperlen * 32
	uint* dev_unaryBitsOri;
	uint* d_PrefixSum;
	uint* d_PrefixSum1;
	uint* d_ZeroIndex;
	uint* d_unary;
	uint* d_result;
	uint* d_mask;
	HANDLE_ERROR(cudaMalloc((void**) &d_mask, 32*sizeof(uint)));
	cudaMemcpy(d_mask, mask, 32*sizeof(uint), cudaMemcpyHostToDevice);

	uint* d_prefix;
	uint* h_prefix = new uint[25000000];
	HANDLE_ERROR(cudaMalloc((void**) &d_prefix, 25000000*sizeof(uint)));

	uint* d_blockprefix;
	uint* h_blockprefix = new uint[25000000];
	HANDLE_ERROR(cudaMalloc((void**) &d_blockprefix, 25000000*sizeof(uint)));

	uint* dev_unarycount;
	HANDLE_ERROR(cudaMalloc((void**) &dev_unarycount, 25000000*sizeof(uint)));
	uint* h_unarycount = new uint[25000000];

	uint* d_output2;
	HANDLE_ERROR(cudaMalloc((void**) &d_output2, 25000000*sizeof(uint)));
	uint* h_output2 = new uint[25000000];

	uint* d_output3;
	HANDLE_ERROR(cudaMalloc((void**) &d_output3, 25000000*sizeof(uint)));
	uint* h_output3 = new uint[25000000];

	uint* d_output4;
	HANDLE_ERROR(cudaMalloc((void**) &d_output4, 25000000*sizeof(uint)));
	uint* h_output4 = new uint[25000000];

	HANDLE_ERROR(cudaMalloc((void**) &dev_unary, 25000000*sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**) &dev_unaryBits, 800000 * 32 *sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**) &dev_unaryBitsOri, 800000 * 32 *sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**) &d_PrefixSum, 800000 * 32 *sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**) &d_PrefixSum1, 800000 * 32 *sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**) &d_ZeroIndex, 25000000*sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**) &d_unary, 25000000*sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**) &d_result, 25000000*sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**) &dev_doc, 25000000*sizeof(uint)));
	HANDLE_ERROR(cudaMalloc((void**) &dev_binary, 25000000*sizeof(uint)));



	if (qlexicon == NULL)
		cout<<"bad";
	else
	{
		while(qlexicon.good())
		{
			qlexicon.read((char*)&quasi, sizeof(Quasilexicon));
			if(!qlexicon.eof())
			{
				cout<<"ListName: "<<quasi.name<<endl<<"List Length: "<<quasi.len<<endl<<"BitWidth: "<<quasi.bw<<endl<<"UpperLength: "<<quasi.upperlen<<endl;
				unsigned int numElements = 32*quasi.upperlen;
				uint num = quasi.upperlen;

				uint* unaryOffset = &buffer[quasi.offset];
				memcpy(packedUnary, unaryOffset, quasi.upperlen*sizeof(uint));
				uint* binaryOffset = &buffer[quasi.upperlen + quasi.offset];
				memcpy(docptr, binaryOffset, quasi.lowerlen * sizeof(uint));

				// Initialize the CUDPP Library
				CUDPPHandle theCudpp;
				CUDPPHandle theCudpp1;
				CUDPPHandle theCudpp2;
				cudppCreate(&theCudpp);
				cudppCreate(&theCudpp1);
				cudppCreate(&theCudpp2);
				CUDPPConfiguration config;
				config.op = CUDPP_ADD;
				config.datatype = CUDPP_INT;
				config.algorithm = CUDPP_SCAN;
				config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
				CUDPPHandle scanplan = 0;
				CUDPPResult res = cudppPlan(theCudpp, &scanplan, config, numElements, 1, 0);
				CUDPPResult res1 = cudppPlan(theCudpp1, &scanplan, config, numElements, 1, 0);
				CUDPPResult res2 = cudppPlan(theCudpp2, &scanplan, config, num, 1, 0);
				cudaMemcpy(dev_unary, packedUnary, quasi.upperlen * sizeof(uint), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_doc, docptr, quasi.lowerlen*sizeof(uint), cudaMemcpyHostToDevice);


				for(int i = 0 ; i<1000; i++){      // This loop is for timing the speed, since the decompression is extremely fast,
								   // we choose to decompression every single list 1000 times, then calculate the average.
				//cout<<i<<endl;
				ti1.start();
				
				BitCount<<<quasi.upperlen/512+1, 512>>>(dev_unary, dev_unarycount, quasi.upperlen);
				cudaDeviceSynchronize();

				ti1.stop();
				t1 = ti1.getTime();
				ti1.reset();
				ti2.start();
				
				cudppScan(scanplan, d_prefix, dev_unarycount, num);
				cudaDeviceSynchronize();
				
				ti2.stop();
				t2 = ti2.getTime();
				ti2.reset();
				ti3.start();
				
			//	cudaMemcpy(h_unarycount, dev_unarycount, quasi.upperlen*sizeof(uint), cudaMemcpyDeviceToHost);

				Blockprefixsum<<<quasi.upperlen/512 + 1, 512>>>(dev_unary, d_prefix, d_blockprefix,  quasi.upperlen);
				cudaDeviceSynchronize();

				ti3.stop();
				t3 = ti3.getTime();
				ti3.reset();
				ti4.start();
				
			//	cudaMemcpy(h_prefix, d_prefix, quasi.upperlen*sizeof(uint), cudaMemcpyDeviceToHost);
				unpack (dev_binary, dev_doc, (uint)quasi.len, (uint)quasi.bw, d_blockprefix, d_result);
				//unpack (dev_binary, dev_doc, (uint)quasi.len, (uint)quasi.bw);
				cudaDeviceSynchronize();
			//	readbits<<<4098, 512>>>(dev_unary, dev_unaryBits, quasi.upperlen);

			//	cudaMemcpy( h_unaryBits, dev_unaryBits, 32*quasi.upperlen*sizeof(uint), cudaMemcpyDeviceToHost);
			//	cudaDeviceSynchronize();
				ti4.stop();
				t4 = ti4.getTime();
				ti4.reset();

				cudaMemcpy(h_blockprefix, d_blockprefix, quasi.len*sizeof(uint), cudaMemcpyDeviceToHost);
				cudaMemcpy( h_result, d_result, quasi.len*sizeof(uint), cudaMemcpyDeviceToHost);
				cudaMemcpy(binaryptr, dev_binary, quasi.len*sizeof(uint), cudaMemcpyDeviceToHost);
				
			// test the decompressed results are the same with original inputs.
/*				
				uint* dat = new uint[quasi.len];
								data.read((char*) & dat[0], quasi.len*sizeof(uint));
								for (int i=0; i<quasi.len; i++) {
									if (h_result[i] != dat[i]) {
									    cout<<i<<" "<<h_result[i] <<"! ="<< dat[i]<<"   "<<h_blockprefix[i]<<" "<<binaryptr[i]<<" "<<h_result[i]<<endl;
										}
									  }
				cout<<"success!"<<endl;
*/
				if ( i >=1){
					total1 += t1;
					total2 += t2;
					total3 += t3;
					total4 += t4;

				}
				float t = t1+t2+t3+t4+t5;
				float speed = 1000*(quasi.len/t)/1000000000;
				total = total1 + total2 + total3 + total4;
				if(i>=1)
				{
					totals += speed;
				}


				}

				// just for creating the charts, nothing useful here.
				double bips = 1000*(quasi.len/(total/99));
				double bips1 = 1000*(quasi.len/(total1/99));
				double bips2 = 1000*(quasi.len/(total2/99));
				double bips3 = 1000*(quasi.len/(total3/99));
				double bips4 = 1000*(quasi.len/(total4/99));
			
				cout<<total1/99<<" "<<total2/99<<" "<<total3/99<<" "<<total4/99<<" "<<total5<<" "<<" "<<total<<endl;
				cout<<"finish"<<endl<<endl;
				cout<<"count 1s: "<<bips1<<endl<<"prefix sum: "<<bips2<<endl<<"gather: "<<bips3<<endl<<"binary+merge: "<<bips4<<endl;
				cout<<"Final: "<<bips<<endl<<endl<<endl;
    			cudppDestroy(theCudpp);
				}
		  }

	}

	cout<<totals<<endl<<totals/99<<endl;
	timer1.stop();
	float totaltime = timer1.getTime();
	cout<<totaltime<<endl;
	
	
	delete[] h_odata;
	delete[] h_result;
	delete[] packedUnary;
	delete[] docptr;
	delete[] binaryptr;
	delete[] h_ZeroIndex;
	delete[] h_unary;
	delete[] h_unaryBits;
	delete[] h_packedUnary;
	delete[] h_docptr;
	delete[] buffer;
	delete[] h_odata1;
	delete[] h_odata2;
	delete[] h_prefix;
	delete[] h_blockprefix;
	delete[] h_unarycount;
    delete[] h_output2;
    delete[] h_output3;
    delete[] h_output4;
    
	cudaFree(dev_unarycount);
	cudaFree(d_blockprefix);
	cudaFree(d_prefix);
    cudaFree(d_PrefixSum);
    cudaFree(d_ZeroIndex);
    cudaFree(d_unary);
    cudaFree(d_result);
    cudaFree(dev_doc);
    cudaFree(dev_binary);
    cudaFree(dev_unary);
    cudaFree(dev_unaryBits);
    cudaFree(d_PrefixSum1);
    cudaFree(dev_unaryBitsOri);
    cudaFree(d_output2);
	cudaFree(d_output3);
 	cudaFree(d_output4);

   
}






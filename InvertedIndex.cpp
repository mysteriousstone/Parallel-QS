/*
 * InvertedIndex.cpp
 *
 *  Created on: Apr 28, 2013
 *      Author: shi
 */

/*
 * This file is a parser. The input is already parsed inverted indexes from TREC-GOV2 dataset. Output is documentID(docID)
 * in binary form.
 * 
 * Input structure: (text inverted index files)
 * 					DocID	Score
 *					1		0.356
 *					13		0.863
 *					25		0.966
 *Output Structure: two files. One data file stores only DocIDs in binary. Since we have thousands of such inverted indexes,
 * 					we have another lexicon file to remember the offsets of each index file.
 *
 */

//build binary inverted index from /team/testdata

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <stdlib.h>
using namespace std;

string path1 = "/home/shi/lists2/";
string path2 = "/home/shi/newlists1/";

struct lexi
{
	char name[20];
	int len;
};
int main()
{
	string line, line1, line2;
	fstream file_list("/home/shi/newlists1/file_list");
	system("ls /home/shi/lists2 >> /home/shi/newlists1/file_list");
	file_list.close();
	ifstream file("/home/shi/newlists1/file_list");
	ofstream data("/home/shi/newlists1/data.dat", ios::binary);
	ofstream lexicon("/home/shi/newlists1/lexicon.dat", ios::binary);
		if (file.is_open())
		{
			while(file.good())
			{
				getline (file, line);
				if(!file.eof())
				{
					string from = path1+line;
					cout<<from<<endl;
					ifstream in(from.c_str());
					if (in == NULL) perror ("Error opening file");
					else
					{
						vector<int> doc;
						lexi lex;
						int b;
						int m = 0;
						while(!in.eof())
						{
							getline(in, line1, ' ');
							getline(in, line2);
							b = atoi(line1.c_str());
							doc.push_back(b);
							m++;

						}
						data.write((char*)&doc[0], m*sizeof(int));
						strcpy(lex.name, line.c_str());
						lex.len = m-1;
						lexicon.write((char*)&lex, sizeof(lexi));
						in.close();
					}
				}
			}
			file.close();
		}
		else
		{
			cout<<"bad"<<endl;
		}
	return 0;
}




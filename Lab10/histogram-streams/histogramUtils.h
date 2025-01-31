#ifndef READTEXTFILE_H_
#define READTEXTFILE_H_

#include <cmath>
#include <fstream>

long getCharsNo(char *filename)
{
	FILE *fp;
	fp = fopen(filename, "r");

	fseek(fp, 0L, SEEK_END);
	long size = ftell(fp);
	fclose(fp);

	return size;
}

void readFile(char *filename, long fileSize, unsigned char *fileBuffer)
{
	FILE *fp;
	fp = fopen(filename, "rb");

	fread(fileBuffer, fileSize, 1, fp);

	fclose(fp);
}

void writeFile(char *filename, unsigned int *histogram, int nBins, int nLetters = 26)
{
	FILE *fp;
	fp = fopen(filename, "w");

	int binWidth = ceil((float)nLetters / nBins);
	if (binWidth == 1)
	{
		for (int i = 0; i < nBins; ++i)
		{
			fprintf(fp, "%c: %u\n", 'a' + i, histogram[i]);
		}
	}
	else
	{
		for (int i = 0; i * binWidth < nLetters; ++i)
		{
			fprintf(fp, "%c-%c: %u\n", 'a' + i * binWidth, (i + 1) * binWidth <= nLetters ? 'a' + (i + 1) * binWidth - 1 : 'z', histogram[i]);
		}
	}

	fclose(fp);
}

#endif /* READTEXTFILE_H_ */

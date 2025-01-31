#include <stdio.h>
#include <stdlib.h>

#define SUCCESS 0
#define NO_FILE 1
#define NO_MEMO 2

#define checkMemory(ptr, str)                             \
    {                                                     \
        if (ptr == NULL)                                  \
        {                                                 \
            printf("%s: Cannot allocate memory.\n", str); \
            return NO_MEMO;                               \
        }                                                 \
    }

int readFile(const char *fileName, int *source, int **edges, int **dest, int **labelSeq, int **labelGlobal, int **labelBlock, int *nodes, int **pFrontierGlobal, int **pFrontierBlock, int *pFrontierTailGlobal, int *pFrontierTailBlock)
{
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        return NO_FILE;
    }

    // read start vertex
    fscanf(fp, "%d", source);

    // read edges table
    int edgesNum;
    fscanf(fp, "%d", &edgesNum);
    *edges = (int *)malloc(edgesNum * sizeof(int));
    checkMemory(*edges, "edges");
    for (int i = 0; i < edgesNum; i++)
    {
        fscanf(fp, "%d ", (*edges) + i);
    }

    // read dest table
    int destNum;
    fscanf(fp, "%d ", &destNum);
    *dest = (int *)malloc(destNum * sizeof(int));
    checkMemory(*dest, "dest");
    for (int i = 0; i < destNum; i++)
    {
        fscanf(fp, "%d ", (*dest) + i);
    }

    // read labels
    fscanf(fp, "%d", nodes);
    *labelSeq = (int *)malloc((*nodes) * sizeof(int));
    checkMemory(*labelSeq, "labelSeq");
    *labelGlobal = (int *)malloc((*nodes) * sizeof(int));
    checkMemory(*labelGlobal, "labelGlobal");
    *labelBlock = (int *)malloc((*nodes) * sizeof(int));
    checkMemory(*labelBlock, "labelBlock");
    for (int i = 0; i < (*nodes); i++)
    {
        fscanf(fp, "%d", (*labelSeq) + i);
        (*labelGlobal)[i] = (*labelSeq)[i];
        (*labelBlock)[i] = (*labelSeq)[i];
    }

    *pFrontierTailGlobal = 1;
    *pFrontierTailBlock = 1;
    *pFrontierGlobal = (int *)malloc((*pFrontierTailGlobal) * sizeof(int));
    checkMemory(*pFrontierGlobal, "pFrontierGlobal");
    *pFrontierBlock = (int *)malloc((*pFrontierTailBlock) * sizeof(int));
    checkMemory(*pFrontierBlock, "pFrontierBlock");
    for (int i = 0; i < *pFrontierTailGlobal; i++)
    {
        (*pFrontierGlobal)[i] = *source;
    }
    for (int i = 0; i < *pFrontierTailBlock; i++)
    {
        (*pFrontierBlock)[i] = *source;
    }

    fclose(fp);
    return SUCCESS;
}

int writeFile(char *fileName, int **label, int *nodes)
{
    FILE *fp;

    fp = fopen(fileName, "w");
    if (fp == NULL)
    {
        return NO_FILE;
    }

    fprintf(fp, "%d\n", *nodes);
    int notVisited = 0;
    for (int i = 0; i < *nodes; i++)
    {
        if (i < 20000)
        {
            fprintf(fp, "%d ", (*label)[i]);
        }
        if ((*label)[i] == -1)
        {
            notVisited++;
        }
    }
    fprintf(fp, "\n");
    fprintf(fp, "%d", notVisited);

    fclose(fp);
    return SUCCESS;
}
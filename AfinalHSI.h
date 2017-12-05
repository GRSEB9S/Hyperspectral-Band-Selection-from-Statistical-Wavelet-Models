/*
Project: UMass Amherst ECE Summer 2014 Research
Chief Editor: Ping Fung (Rising ECE Senior)
Advisors: Professor Marco Duarte, Professor Mario Parente
More Advisors: Siwei Feng, and others

Note: AfinalHSI.h is designed to work with AfinalHSI.cpp.
      The two files have to be in the same folder.
      AfinalHSI.h outlines all of the functions of the code.
*/

#ifndef HEADER
#define HEADER

struct tNhmc{
double ****transitions;
//(nstates,nstates,lengthseq,bands);
double **initprobs;
//nstates, bands
double ***gaussmu;
//nstates,lengthseq,bands);
double ***gaussSigma;

double ***newpS;
};

struct emChains{
double ***ES;   //[M,M,L][nstates,lseq]
double *POS;    //[/[M,1]
double **MU;    //]M,L]
double **SI;    //[M,L]
double **pS;
};

struct IND{
double *epsi;
double *mui;
double *vari;
}; //dim nmix or M

struct udReturn{
double ***gamma;
double ***alpha;
double ***beta;
double ***btpni;
//m,l,n
double *lk;
};

void M2C(double inM[], double ***cMat,int dims);
void C2M2D(double **cMat,   double outM[],int *dims);
void C2M3D(double ***cMat,  double outM[],int *dims);
void C2M4D(double ****cMat, double outM[],int *dims);

double* createM1(int x);
double** createM2(int r, int c);
double*** createM3(int x, int y, int z);
double**** createM4(int a, int b, int c, int d);
void free2d(double**M, int r);
void free3d(double*** M, int x, int y);

double **sum32(int ***M, int x, int y, int z, int dim);
double ** squeeze32(int **M, int x, int y, int z, int dim);


tNhmc train_nhmc(double ***wmat);

emChains trainEM_chains(double **x,
               int M, int maxIter);
      //M is nstates
emChains startPoints(double **x, int M);
//x is wltdata [lengthseq, thirdD]
IND indmixmod(double *dat, int nmix,
              int nIter, int zm);
//nmix is nstates
IND initem(double *dat,int nmix);
//data is thirdD dim
udReturn udChain(double **x, double ***ES, double *POS,
                 double **MU, double **SI, double scl);
double* gaussian(double *x, double* mui, double* vari);
//returns f (nstates)

#endif // HEADER

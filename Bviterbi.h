/*
Project: UMass Amherst ECE Summer 2014 Research
Chief Editor: Ping Fung (Rising ECE Senior)
Advisors: Professor Marco Duarte, Professor Mario Parente
More Advisors: Siwei Feng, and others

Note: Bviterbi.h is designed to work with Bviterbi.cpp.
      The two files have to be in the same folder.
      Bviterbi.h outlines all of the functions of the code.
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

void M2C2D(double inM[], double **oMat, int *dims);
void M2C3D(double inM[], double ***oMat,int *dims);
void M2C4D(double inM[], double ****oMat, int *dims);
void C2M3D(double ***cMat,  double outM[],int *dims);

double* createM1(int x);
double** createM2(int r, int c);
double*** createM3(int x, int y, int z);
double**** createM4(int a, int b, int c, int d);
void free2d(double**M, int r);
void free3d(double*** M, int x, int y);

double ***nhmc_labels(double ***wmat, tNhmc train);

double **viterbi_chain(double **x, double ***ES, double *POS,
                        double **MU, double **SI, int scl);

double* gaussian(double *x, double* mui, double* vari);
//returns f (nstates)

double oGaussian(double x, double mui, double vari);
double argmax(double *x);
#endif // HEADER

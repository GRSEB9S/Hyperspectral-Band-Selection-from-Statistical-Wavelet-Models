/*
Project: UMass Amherst ECE Summer 2014 Research
Chief Editor: Ping Fung (Rising ECE Senior)
Advisors: Professor Marco Duarte, Professor Mario Parente
More Advisors: Siwei Feng, and others

Note: Bviterbi.cpp must be in the same folder as Bviterbi.h

Program Input: A 3D matrix that results from the wavelet transform of a 2D
                  matrix that represents multiple spectra: (wmat)
               The dimensions of the matrix and the number of states:
                  (lengthseq, bands, thirdD, nstates)
               The output matrices from the training (EM) algorithm:
                  (transitions, initprobs, gaussmu, gaussSigma)

Program Outputs: State labels: (lengthseq, bands, thirdD)
*/

#include <stdio.h>
#include <iostream>
#include <stdlib.h> //for malloc
#include <cmath>
#include "Bviterbi.h"
#include "mex.h"

using namespace std;
int lengthseq, bands, thirdD, nstates;

void M2C2D(double inM[], double **oMat, int *dims)
{
   for (int i=0; i<dims[0]; i++)
      for (int j=0; j<dims[1]; j++)
         oMat[i][j]=inM[i+j*dims[0]];
}

void M2C3D(double inM[], double ***oMat, int *dims)
{
   for (int i=0; i<dims[0]; i++)
      for (int j=0; j<dims[1]; j++)
         for (int k=0; k<dims[2]; k++)
         oMat[i][j][k]=inM[i+j*dims[0]+k*dims[0]*dims[1]];
}

void M2C4D(double inM[], double ****oMat, int *dims)
{
   for (int i=0; i<dims[0]; i++)
      for (int j=0; j<dims[1]; j++)
         for (int k=0; k<dims[2]; k++)
            for (int l=0; l<dims[3]; l++)
            oMat[i][j][k][l]=inM[i+j*dims[0]+k*dims[0]*dims[1]+l*dims[0]*dims[1]*dims[2]];

}

void C2M3D(double ***cMat, double outM[], int *dims)
{
   for (int i=0; i<dims[0]; i++)
      for (int j=0; j<dims[1]; j++)
         for (int k=0; k<dims[2]; k++)
         outM[i+j*dims[0]+k*dims[0]*dims[1]]=cMat[i][j][k];
}


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
   double *inM, *inM2, ***cMat;
   double *iT, *iI, *iM, *iS;
   tNhmc tM;
   double *outL1;
   int *dims, *tDims, *iDims, *gDims;

   inM  = mxGetPr(prhs[0]); inM2 = mxGetPr(prhs[1]);
   iT  = mxGetPr(prhs[2]);  iI = mxGetPr(prhs[3]);
   iM  = mxGetPr(prhs[4]);  iS = mxGetPr(prhs[5]);

   int *locat0, *locat1, *locat2, *locat3;
   locat0=&lengthseq; locat1=&bands;
   locat2=&thirdD; locat3=&nstates;

   *locat0=(int)inM2[0]; *locat1=(int)inM2[1];
   *locat2=(int)inM2[2]; *locat3=(int)inM2[3];

   dims=(int *)malloc(5*sizeof(int));
   tDims=(int *)malloc(5*sizeof(int));
   iDims=(int *)malloc(5*sizeof(int));
   gDims=(int *)malloc(5*sizeof(int));

   dims[0]=lengthseq; dims[1]=bands ; dims[2]=thirdD;
   tDims[0]=nstates; tDims[1]=nstates; tDims[2]=lengthseq; tDims[3]=bands;
   iDims[0]=nstates; iDims[1]=bands;
   gDims[0]=nstates; gDims[1]=lengthseq; gDims[2]=bands;

   cMat=createM3(dims[0],dims[1],dims[2]);
   tM.transitions=createM4(tDims[0],tDims[1],tDims[2],tDims[3]);
   tM.initprobs = createM2(iDims[0],iDims[1]);
   tM.gaussmu   = createM3(gDims[0],gDims[1],gDims[2]);
   tM.gaussSigma= createM3(gDims[0],gDims[1],gDims[2]);

   /* Create matrix for the return argument. */
   plhs[0]=mxCreateNumericArray(3, dims , mxDOUBLE_CLASS, mxREAL);

  /* Assign pointers to each output. */
   outL1 = mxGetPr(plhs[0]);

   M2C3D(inM, cMat, dims);
   M2C4D(iT, tM.transitions, tDims);
   M2C2D(iI, tM.initprobs, iDims);
   M2C3D(iM, tM.gaussmu, gDims);
   M2C3D(iS, tM.gaussSigma, gDims);

   double ***nLabels;
   nLabels=nhmc_labels(cMat,tM);

   C2M3D(nLabels, outL1, dims);
}


double* createM1(int x)
{
   double *M;
   M=(double *)malloc(x*sizeof(double));
   for (int i=0; i<x; i++)
      M[i]=0;
   return M;
}

double** createM2(int r, int c)
{
   double** M;
   M = (double **) malloc (r*sizeof(double*));
   for (int i=0; i<r; i++){
      M[i]=(double *) malloc (c*sizeof(double));
      for (int j=0; j<c; j++)
         M[i][j]=0;
   }
   return M;
}

double*** createM3(int x, int y, int z)
{
   double*** M;
   M = (double ***) malloc (x*sizeof(double**));
   for (int i=0; i<x; i++){
      M[i]=(double **) malloc (y*sizeof(double*));
      for (int j=0; j<y; j++)
         M[i][j]=(double *) malloc (z*sizeof(double));
   }
    return M;
}

double**** createM4(int a, int b, int c, int d)
{
    double**** tempM;
    tempM = (double ****) malloc (a*sizeof(double***));
    for (int i=0; i<a; i++){
            tempM[i]= createM3(b,c,d);
        }
    return tempM;
}


void free2d(double** M, int r)
{
    for (int i=0; i < r; i++)
        free(M[i]);
    free(M);
}

void free3d(double*** M, int x, int y)
{
    for (int i = 0; i < x; i++)
        free2d(M[i], y);
    free(M);
}



double ***nhmc_labels(double ***wmat, tNhmc train)
{
double ***qp, **wltdata, **tempM;
   qp=createM3(lengthseq, thirdD, bands);
   wltdata=createM2(lengthseq, thirdD);

double ***tempT, *tempI, **tempMU, **tempS;
   tempT=createM3(nstates, nstates, lengthseq);
   tempI=createM1(nstates);
   tempMU=createM2(nstates, lengthseq);
   tempS=createM2(nstates, lengthseq);

   for (int wlt=0; wlt<bands; wlt++){
      for (int i=0; i<lengthseq; i++)
         for (int k=0; k<thirdD; k++)
            wltdata[i][k]=wmat[i][wlt][k];

      for (int i=0; i<nstates; i++){
         tempI[i]=train.initprobs[i][wlt];
         for (int j=0; j<lengthseq; j++){
            tempMU[i][j]= train.gaussmu[i][j][wlt];
            tempS[i][j]= train.gaussSigma[i][j][wlt];
            for (int k=0; k<nstates; k++)
               tempT[i][k][j]=train.transitions[i][k][j][wlt];
         }
      }

      tempM=viterbi_chain(wltdata,tempT, tempI, tempMU, tempS,0);
      for (int i=0; i<lengthseq; i++)
         for (int j=0; j<thirdD; j++)
            qp[i][j][wlt]=tempM[i][j];

   }
double ***qp1;
qp1=createM3(lengthseq, bands, thirdD);

for (int i=0; i<lengthseq; i++)
   for (int j=0; j<bands; j++)
      for (int k=0; k<thirdD; k++){
         qp1[i][j][k]=qp[i][k][j];
     //    if (qp1[i][j][k]<=10 && qp1[i][j][k]>=1)
        //    qp1[i][j][k]=qp1[i][j][k]-1;
      }//commented statements are unnecessary because of cpp indexing.


free3d(qp, lengthseq, thirdD);
free3d(tempT, nstates, nstates);
free(tempI);
free2d(tempMU, nstates);
free2d(tempS, nstates);

   return qp1;
}


double **viterbi_chain(double **x, double ***ES, double *POS,
                        double **MU, double **SI, int scl)
{
   int N= thirdD;
   int M= nstates;
   int L= lengthseq;
double ***delta, ***psi, **qp;
   delta=createM3(M,L,N);
   psi=createM3(M,L,N);
   qp=createM2(L,N);

double *tempX, *tempM, *tempS;
   tempX=createM1(nstates);
   tempM=createM1(nstates);
   tempS=createM1(nstates);

//intialization
double *f1;
   for (int ii=0; ii<N; ii++){
      for (int i=0; i<nstates; i++){
         tempX[i]=x[0][ii];
         tempM[i]=MU[i][0];
         tempS[i]=SI[i][0];
      }
      f1=gaussian(tempX, tempM, tempS);

      if (scl==0){
         for (int i=0; i<M; i++)
            delta[i][0][ii]=POS[i]* f1[i];
      }
      else{
         for (int i=0; i<M; i++)
            delta[i][0][ii]=log(POS[i])+log(f1[i]);
      }
free(f1);
      for (int i=0; i<M; i++)
         psi[i][0][ii]=0;
   }


// recursion
double f;
for (int ll = 1; ll<L; ll++){
   for (int ii = 0; ii<N; ii++){
      for (int jj = 0; jj<M; jj++){
         f = oGaussian(x[ll][ii], MU[jj][ll], SI[jj][ll]);

         if (scl == 0){
            double maxN=delta[0][ll-1][ii]*ES[jj][0][ll];
            for (int i=0; i<M; i++)
               if (maxN<delta[i][ll-1][ii]*ES[jj][i][ll])
                  maxN= delta[i][ll-1][ii]*ES[jj][i][ll];
            delta[jj][ll][ii]=maxN*f;
         }
         else{
            double maxN=0;
            for (int i=0; i<M; i++)
               if (maxN<delta[i][ll-1][ii]*log(ES[jj][i][ll]))
                  maxN= delta[i][ll-1][ii]*log(ES[jj][i][ll]);
               delta[jj][ll][ii]=maxN+log(f);
         }

         if (scl == 0){
            for (int i=0; i<nstates; i++)
               tempM[i]=delta[i][ll-1][ii]*ES[jj][i][ll];
            psi[jj][ll-1][ii] = argmax(tempM);
         }
         else{
            for (int i=0; i<nstates; i++)
               tempM[i]=delta[i][ll-1][ii]+log(ES[jj][i][ll]);
            psi[jj][ll-1][ii] = argmax(tempM);
         }
      }
   }
}//end of triple for loop


// termination
// at level L
for (int ii = 0; ii<N; ii++){
   for (int i=0; i<nstates; i++)
      tempM[i]=delta[i][L-1][ii];
   qp[L-1][ii] = argmax(tempM);
}


// path (state sequence) backtracking
for (int ll = (L-2); ll>=0; ll--)
   for (int ii = 0; ii<N; ii++)
      for (int i=0; i<nstates; i++)
         qp[ll][ii] = psi [(int)qp[ll+1][ii]] [ll] [ii];

free(tempX); free(tempM); free(tempS);
return qp;
}



double* gaussian(double *x, double* mui, double* vari)//CHECK
{
   double num, denom;
   int ind=0;
   double *f;
   f =(double *)malloc(nstates*sizeof(double));

   for (int i=0; i<nstates; i++){
      if (vari[i]<1e-5)
         vari[i]=1e-5;
      num=exp(-pow(x[i]-mui[i],2)/(2*vari[i]));
      denom=sqrt(vari[i]*2*3.14159265);
      f[i]=num/denom;
   }

   for (int i=0; i<nstates; i++){
      if (f[i]<1e-20)
         f[i]=1e-20;
   }
   return f;
}


double oGaussian(double x, double mui, double vari)
{
   double num, denom, f;

   if (vari<1e-5)
      vari=1e-5;
   num=exp(-pow(x-mui,2)/(2*vari));
   denom=sqrt(vari*2*3.14159265);
   f=num/denom;
   if (f<1e-20)
      f=1e-20;

   return f;
}


// quick argmax function
// x - matrix to find argmax of
// dim - dimension to look along
double argmax(double *x)
{
   double am=0;
   double maxN=x[0];
   for (int i=0; i<nstates; i++)
      if (x[i]>maxN){
         maxN=x[i];
         am=i;
      }
   return am;
}


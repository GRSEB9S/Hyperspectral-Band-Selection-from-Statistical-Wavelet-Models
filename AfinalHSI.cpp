/*
Project: UMass Amherst ECE Summer 2014 Research
Chief Editor: Ping Fung (Rising ECE Senior)
Advisors: Professor Marco Duarte, Professor Mario Parente
More Advisors: Siwei Feng, and others

Note: AtrainHSI.cpp must be in the same folder as AtrainHSI.h

Program Input: A 3D matrix that results from the wavelet transform
               of a 2D matrix that represents multiple spectra.

Program Outputs: Transition Matrix:  (nstates,nstates,lengthseq,bands)
                 GaussMu Matrix: (nstates,lengthseq,bands)
                 GaussSigma Matrix: (nstates,lengthseq,bands)
                 InitialProbability Matrix:(nstates,bands)
                 newpS: (nstates, lengthseq, bands)
*/

#include <stdio.h>
#include <iostream>
#include <stdlib.h> //for malloc
#include <cmath>
#include "AfinalHSI.h"
#include "mex.h"

using namespace std;
int lengthseq, bands, thirdD, nstates;

void M2C(double inM[], double ***cMat, int *dims)
{
   for (int i=0; i<dims[0]; i++)
      for (int j=0; j<dims[1]; j++)
         for (int k=0; k<dims[2]; k++)
         cMat[i][j][k]=inM[i+j*dims[0]+k*dims[0]*dims[1]];
}

void C2M2D(double **cMat, double outM[], int *dims)
{
   for (int i=0; i<dims[0]; i++)
      for (int j=0; j<dims[1]; j++)
         outM[i+j*dims[0]]=cMat[i][j];
}

void C2M3D(double ***cMat, double outM[], int *dims)
{
   for (int i=0; i<dims[0]; i++)
      for (int j=0; j<dims[1]; j++)
         for (int k=0; k<dims[2]; k++)
         outM[i+j*dims[0]+k*dims[0]*dims[1]]=cMat[i][j][k];
}

void C2M4D(double ****cMat, double outM[], int *dims)
{
   for (int i=0; i<dims[0]; i++)
      for (int j=0; j<dims[1]; j++)
         for (int k=0; k<dims[2]; k++)
            for (int l=0; l<dims[3]; l++)
               outM[i+j*dims[0]+k*dims[0]*dims[1]+l*dims[0]*
               dims[1]*dims[2]]=cMat[i][j][k][l];
}


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
   double *inM, *inM2, ***cMat;
   double *outM1, *outM2, *outM3, *outM4, *outM5;
   int *dims, *tDims, *iDims, *gDims;

   inM  = mxGetPr(prhs[0]);
   inM2 = mxGetPr(prhs[1]);

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
   tDims[0]=nstates; tDims[1]=nstates;
   tDims[2]=lengthseq; tDims[3]=bands;
   iDims[0]=nstates; iDims[1]=bands;
   gDims[0]=nstates; gDims[1]=lengthseq; gDims[2]=bands;

   cMat=createM3(dims[0],dims[1],dims[2]);

   /* Create matrix for the return argument. */
   plhs[0]=mxCreateNumericArray(4, tDims, mxDOUBLE_CLASS, mxREAL);
   plhs[1]=mxCreateNumericArray(2, iDims, mxDOUBLE_CLASS, mxREAL);
   plhs[2]=mxCreateNumericArray(3, gDims, mxDOUBLE_CLASS, mxREAL);
   plhs[3]=mxCreateNumericArray(3, gDims, mxDOUBLE_CLASS, mxREAL);
   plhs[4]=mxCreateNumericArray(3, gDims , mxDOUBLE_CLASS, mxREAL);

  /* Assign pointers to each output. */
   outM1 = mxGetPr(plhs[0]);
   outM2 = mxGetPr(plhs[1]);
   outM3 = mxGetPr(plhs[2]);
   outM4 = mxGetPr(plhs[3]);
   outM5 = mxGetPr(plhs[4]);

   M2C(inM, cMat, dims);

   tNhmc tM;
   tM=train_nhmc(cMat);

   C2M4D(tM.transitions,outM1, tDims);
   C2M2D(tM.initprobs,  outM2, iDims);
   C2M3D(tM.gaussmu,    outM3, gDims);
   C2M3D(tM.gaussSigma, outM4, gDims);
   C2M3D(tM.newpS, outM5, gDims);
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

double** sum32(double ***M, int x, int y, int z, int dim)
{
   double **newM;
   switch (dim){
      case 1:
         newM=createM2(y,z);
         for (int j=0; j<y; j++)
            for (int k=0; k<z; k++){
               newM[j][k]=0;
               for (int i=0; i<x; i++)
                  newM[j][k]=newM[j][k]+M[i][j][k];
            }
         break;
      case 2:
         newM=createM2(x,z);
         for (int i=0; i<x; i++)
            for (int k=0; k<z; k++){
               newM[i][k]=0;
               for (int j=0; j<y; j++)
                  newM[i][k]=newM[i][k]+M[i][j][k];
            }
         break;
      case 3:
         newM=createM2(x,y);
         for (int i=0; i<x; i++)
            for (int j=0; j<y; j++){
               newM[i][j]=0;
               for (int k=0; k<z; k++)
                  newM[i][j]=newM[i][j]+M[i][j][k];
            }
         break;
    }

    return newM;
}

double** squeeze32(int **M, int x, int y, int z, int dim)
{//not useful
    double **newM;
    return newM;
}


tNhmc train_nhmc(double ***wmat)
{
tNhmc train;
emChains chains;
double **wltdata= createM2(lengthseq, thirdD);

train.transitions=createM4(nstates,nstates,lengthseq,bands);
train.gaussmu=createM3(nstates,lengthseq,bands);
train.gaussSigma=createM3(nstates,lengthseq,bands);
train.initprobs=createM2(nstates,bands);
train.newpS=createM3(nstates, lengthseq, bands);

for (int wlt=0; wlt<bands; wlt++){
    //extract data for each wavelength
    for (int i=0; i<lengthseq; i++)
        for (int j=0; j<thirdD; j++){
            wltdata[i][j]=wmat[i][wlt][j];
        }
    chains=trainEM_chains(wltdata, nstates,30);
//num of iterations can change.
    for (int i=0; i<nstates; i++){
        train.initprobs[i][wlt]=chains.POS[i];
        for (int j=0; j<lengthseq; j++){
            train.gaussmu[i][j][wlt]=chains.MU[i][j];
            train.gaussSigma[i][j][wlt]=chains.SI[i][j];
            train.newpS[i][j][wlt]=chains.pS[i][j];
            for (int k=0; k<nstates; k++)
                train.transitions[i][k][j][wlt]=chains.ES[i][k][j];
    }}//for loops

    free3d(chains.ES, nstates, nstates);
    free(chains.POS);
    free2d(chains.MU, nstates);
    free2d(chains.SI, nstates);
}

return train;
}//train function


emChains trainEM_chains(double **x, int M,int maxIter)
{
   emChains rChain;

   // number of data points and resolutions
   int N = thirdD;  // they got switched.
   int L = lengthseq;

   // get starting points for model parameters
   rChain= startPoints(x, M); //M is nstate

   // loop until we have converged
   rChain.pS=createM2(M,L);
double ***pSio, ***ESo, **SIo, **pSo;
   pSio=createM3(M,L,N);  ESo= createM3(M,M,L);
   SIo= createM2(M,L);  pSo= createM2(M,L);
double ***pSiSp, **btl, **btpil, **alpl, **meanM;
   pSiSp= createM3(M,M,N); btl  = createM2(M,N); btpil= createM2(M,N);
   alpl = createM2(M,N);  meanM= createM2(M,M);
double **tMat, **tSum, *tMean;
   tMat =createM2(M,N);  tSum =createM2(M,N);  tMean=createM1(N);

   int converged = 0;
   int iter = 0;


while (converged == 0){
   // "E step"
   udReturn uDown;
   uDown= udChain(x, rChain.ES, rChain.POS, rChain.MU, rChain.SI, 1);
   //pSi is gamma?

   for (int i=0; i<M; i++)
      for (int j=0; j<L; j++)
         for (int k=0; k<N; k++)
            pSio[i][j][k]=uDown.gamma[i][j][k];

   // "M step"
   for (int i=0; i<M; i++)
      for (int j=0; j<M; j++)
         for (int k=0; k<L; k++){
            ESo[i][j][k]=rChain.ES[i][j][k];
            SIo[j][k]=rChain.SI[j][k];
            pSo[j][k]=rChain.pS[j][k];
         }
   // update the scale constant mixture probabilities

   for (int i=0; i<M; i++)
      for (int j=0; j<L; j++){
         rChain.pS[i][j]=0;
         for (int k=0; k<N; k++)
            rChain.pS[i][j]=rChain.pS[i][j]+uDown.gamma[i][j][k]/N;
      }

   for (int ll = 0; ll<L ; ll++){

      for (int i=0; i<M; i++)
         for (int k=0; k<N; k++){
            btl[i][k] = uDown.beta[i][ll][k];
            btpil[i][k]=uDown.btpni[i][ll][k];
         }
      // calculate joint pmf between child and parent states to
      //update the transition matrix
      if (ll > 0){
         for (int i=0; i<M; i++)
            for (int k=0; k<N; k++)
               alpl[i][k] = uDown.alpha[i][ll-1][k];
         for (int mm = 0; mm<M; mm++)
            for (int nn = 0; nn<M; nn++)
               for (int k=0; k<N; k++){
               pSiSp[mm][nn][k]= ESo[mm][nn][ll]*btl[mm][k]*
                                 alpl[nn][k]*btpil[nn][k];
               }

double **tSum1, *tSum2;
         tSum1= sum32(pSiSp,M,M,N,1);
         tSum2=createM1(N);

         for (int k=0; k<N; k++){
            tSum2[k]=0;
            for (int j=0; j<M; j++)
               tSum2[k]=tSum2[k]+tSum1[j][k];
         }

         for (int i=0; i<M; i++)
            for (int j=0; j<M; j++)
               for (int k=0; k<N; k++)
               pSiSp[i][j][k]=pSiSp[i][j][k]/tSum2[k];
free2d(tSum1, M); free(tSum2);

         for (int i=0; i<M; i++){
            for (int j=0; j<M; j++){
               meanM[i][j]=0;
               for (int k=0; k<N; k++){
                  meanM[i][j]=meanM[i][j]+pSiSp[i][j][k]/N;
               }
               rChain.ES[i][j][ll]=meanM[i][j]/rChain.pS[j][ll-1];
            }
         }
      }// if

      // update the mixture means
      for (int i=0; i<M; i++)
         rChain.MU[i][ll] = 0;

      // update the mixture variances

      for (int i=0; i<M; i++){
         for (int k=0; k<N; k++){
            tMat[i][k]=uDown.gamma[i][ll][k];
            tSum[i][k]=tMat[i][k]*pow((x[ll][k]-rChain.MU[i][ll]),2);
      }  }

      for (int i=0; i<M; i++){
         tMean[i]=0;
         for (int k=0; k<N; k++){
            tMean[i]=tMean[i]+tSum[i][k]/N;
         }
         rChain.SI[i][ll]=tMean[i]/rChain.pS[i][ll];
      }

   }

   // update the initial state distribution
   for (int i=0; i<M; i++)
      rChain.POS[i] = rChain.pS[i][0];

   // test for convergence
   iter = iter + 1;
   if (iter >= maxIter){
      converged = 1;
double **err1, **err2, **err3;
      err1=createM2(M,L);
      err2=createM2(M,L);
      err3=createM2(M,L);

      for (int i=0; i<M; i++){
         for (int j=0; j<L; j++){
            err1[i][j] = abs(rChain.SI[i][j]-SIo[i][j]);
            err2[i][j] = abs(rChain.pS[i][j]-pSo[i][j]);
            for (int k=0; k<N; k++)
               err3[i][j] = abs(uDown.gamma[i][j][k]-pSio[i][j][k]);
      }  }
free2d(err1,M); free2d(err2,M); free2d(err3,M);
   }
free3d(uDown.alpha,M,L); free3d(uDown.beta, M,L);
free3d(uDown.btpni,M,L); free3d(uDown.gamma,M,L);

}  // while loop

return rChain;

}


// function to pick starting points for the EM alg
// by fitting independent mixture models
emChains startPoints(double **x, int M)
{
 //M is nstate
emChains rPoint;
int L = lengthseq;

// allocate space for model parameters
rPoint.MU = createM2(M,L);
rPoint.SI = createM2(M,L);
rPoint.POS=createM1(M);
rPoint.ES=createM3(M,M,L);

// pick starting points for the EM alg by
//fitting independent mixture models
for (int ll = 0; ll<L; ll++){
   IND start;
   start=indmixmod(x[ll],M,30,1);

   for (int i=0; i<M; i++){
      rPoint.MU[i][ll]=start.mui[i];
      rPoint.SI[i][ll]=start.vari[i];
   }
free(start.mui); free(start.vari); free(start.epsi);
}

for (int i=0; i<M; i++){
   rPoint.SI[i][0]=rPoint.SI[i][1];
   rPoint.POS[i]=1/(double)nstates;
   for (int j=0; j<M; j++)
      for (int k=0; k<L; k++)
         rPoint.ES[i][j][k]=1/(double)nstates;
}

return rPoint;
}


IND indmixmod(double *dat, int nmix, int niter, int zm)
{
//nmix is nstates
   IND indR;
   int L = thirdD;
   indR=initem(dat, nmix);

   // default is zero mean
double **datmat, **varimat, **muimat, **epsimat;
   datmat= createM2(L,nmix);
   varimat=createM2(L,nmix);
   muimat= createM2(L,nmix);
   epsimat=createM2(L,nmix);

   for (int i=0; i<thirdD; i++)
      for (int j=0; j<nmix; j++){
         datmat[i][j]=dat[i];
         varimat[i][j]=indR.vari[j];
         muimat [i][j]=indR.mui[j];
         epsimat[i][j]=indR.epsi[j];
      }

double **expV, **pst;
   expV=createM2(L,nmix);
   pst =createM2(L,nmix);

   for (int iter=0; iter<niter; iter++){
      for (int i=0; i<L; i++)
         for (int j=0; j<nmix; j++){
            expV[i][j]=pow(datmat[i][j]-muimat[i][j],2)/(2*varimat[i][j]);
            pst[i][j]=exp(-expV[i][j])/sqrt(varimat[i][j])*epsimat[i][j];
         }

double *scale, **scalemat;
      scale=createM1(L);
      scalemat=createM2(L,nmix);

      for (int i=0; i<L; i++){
         scale[i]=0;
         for (int j=0; j<nmix; j++)
            scale[i]=scale[i]+pst[i][j];
         scale[i]=scale[i]+2.22e-16;
      }

      for (int i=0; i<L; i++)
         for (int j=0; j<nmix; j++){
            scalemat[i][j]=scale[i];
            pst[i][j]=pst[i][j]/scalemat[i][j];
         }
         //keyboard

free(scale); free2d(scalemat,L);

      for (int j=0; j<nmix; j++){
         indR.epsi[j]=0;
         for (int i=0; i<L; i++)
            indR.epsi[j]=indR.epsi[j]+pst[i][j]/L;
         indR.epsi[j]=indR.epsi[j]+2.22e-16;
      }


      for (int i=0; i<L; i++)
         for (int j=0; j<nmix; j++)
            epsimat[i][j] = indR.epsi[j];

      // For denoising, constraint mui to be 0 so skip next statement
      if (zm != 0)
         for (int j=0; j<nmix; j++)
            indR.mui[j] = 0;
      else
         for (int j=0; j<nmix; j++){
            double mean=0;
            for (int i=0; i<L; i++)
               mean= mean+pst[i][j]*datmat[i][j]/L;
            indR.mui[j] =mean/indR.epsi[j];
         }

      for (int i=0; i<L; i++)
         for (int j=0; j<nmix; j++)
            muimat[i][j]=indR.mui[j];

      //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
double **tempM, *meanM;
      tempM=createM2(L,nmix);
      meanM=createM1(nmix);

      for (int j=0; j<nmix; j++){
         meanM[j]=0;
         for (int i=0; i<L; i++){
            tempM[i][j]=pst[i][j]*pow(datmat[i][j]-muimat[i][j],2);
            meanM[j]=meanM[j]+tempM[i][j]/L;
         }
         indR.vari[j]=meanM[j]/indR.epsi[j];
      }

free2d(tempM, L);  free(meanM);

      // Constrain variances > 0, so no singular solutions
      // Minimum value depends on application
double vartol = 1e-5;

      for (int j=0; j<nmix; j++){
         if (indR.vari[j]>vartol&&indR.vari[j]<=vartol)
            indR.vari[j]=indR.vari[j]+vartol;

         if (indR.vari[j]>vartol&&indR.vari[j]>vartol)
            indR.vari[j]=indR.vari[j];

         if (indR.vari[j]<vartol&&indR.vari[j]<=vartol)
            indR.vari[j]=vartol;

         if (indR.vari[j]<vartol&&indR.vari[j]>vartol)
            indR.vari[j]=0;
      }

      for (int i=0; i<L; i++)
         for (int j=0; j<nmix; j++)
            varimat[i][j]=indR.vari[j];

   //keyboard
   }

   // sort the states by size of variance

   for (int x=nmix-1; x>0; x--){
      for (int i=0; i<x; i++){
double temp;
         if (indR.vari[i]>indR.vari[i+1]){
            temp=indR.vari[i];
            indR.vari[i]=indR.vari[i+1];
            indR.vari[i+1]=temp;

            temp=indR.epsi[i];
            indR.epsi[i]=indR.epsi[i+1];
            indR.epsi[i+1]=temp;

            temp=indR.mui[i];
            indR.mui[i]=indR.mui[i+1];
            indR.mui[i+1]=temp;
         }
      }
   }
   //keyboard

free2d(datmat, L); free2d(varimat,L);
free2d(muimat, L); free2d(epsimat,L);
free2d(expV, L);   free2d(pst, L);

   return indR;
}


// initialization
//function [p, mui, sig]
IND initem(double *dat, int nmix)
{
   IND initR;
   initR.epsi=createM1(nmix);
   initR.mui =createM1(nmix);
   initR.vari=createM1(nmix);

   double maxdat=dat[0] ;
   double mindat=dat[0] ;

   for (int i=0; i<thirdD; i++){
      if (maxdat<dat[i])
         maxdat=dat[i];
      if (mindat>dat[i])
         mindat=dat[i];
   }

double *div;
   div=createM1(nmix+1);

   for (int i=0; i<nmix+1; i++)
      div[i] = mindat+ i*(maxdat-mindat)/nmix;

   for (int i=0; i<nmix; i++)
      initR.mui[i]=(div[i+1]-div[i])/2+div[i];

   div[nmix] = div[nmix] + 1e-3;

   if (nmix < 2){
      for (int i=0; i<nmix; i++)
         initR.vari[i] = div[i+1]-div[i];
   }
   else{
      for (int i=0; i<nmix; i++)
         initR.vari[i] = pow(initR.mui[1] - initR.mui[0],2);
   }

   for (int ii = 0; ii<nmix; ii++){
      int sum=0;
      for (int i=0; i<thirdD; i++){
         if (dat[i]>=div[ii]&&dat[i]<div[ii+1])
            sum++;
      }

      initR.epsi[ii] = (double)sum/thirdD;
   }

free(div);
   return initR;
}


udReturn udChain(double **x, double ***ES, double *POS,
                 double **MU, double **SI, double scl)
{
udReturn udRet;
// number of data "nodes", number of levels in tree, and number of states
int N = thirdD;
int L = lengthseq;
int M = nstates;

// upward (beta) and downward (alpha) parameters
udRet.gamma = createM3(M,L,N);
udRet.alpha = createM3(M,L,N);
udRet.beta  = createM3(M,L,N);
udRet.btpni = createM3(M,L,N);
double ***betaipi= createM3(M,L,N);

// scaling coefficients, not in the wavelet sense,
// but in the preventing arithmetic underflow sense
double **c = createM2(L,N);

// ind. likelihood values for each wavelet coefficient in each state
double ***f = createM3(M,L,N);

// "up" step

// intialization
double *tempG, *tempRep, *tempMU, *tempSI;
tempRep=createM1(M);//nstates
tempMU=createM1(M);
tempSI=createM1(M);

for (int ii = 0; ii<thirdD; ii++){
   for (int i=0; i<nstates; i++){
      tempRep[i]= x[L-1][ii];
      tempMU[i] = MU[i][L-1];
      tempSI[i] = SI[i][L-1];
   }
   tempG = gaussian(tempRep, tempMU, tempSI);

   c[L-1][ii]=0;
   for (int i=0; i<M; i++){
      f[i][L-1][ii] = tempG[i];
      udRet.beta[i][L-1][ii] = tempG[i];
      c[L-1][ii]=c[L-1][ii]+tempG[i];
   }
free(tempG);

   if (scl!= 0){
      for (int i=0; i<M; i++)
         udRet.beta[i][L-1][ii]=udRet.beta[i][L-1][ii]/c[L-1][ii];
   }
}

double **squeezeM;
squeezeM=createM2(M,N); //es m*m; nstates,thirdD

for (int i=0; i<M; i++)
    for (int k=0; k<N; k++)
        squeezeM[i][k]=udRet.beta[i][L-1][k];


for (int i=0; i<M; i++){
   for (int j=0; j<N; j++){
      betaipi[i][L-1][j]=0;  //makes sure it's zero.
      for (int k=0; k<M; k++){
         betaipi[i][L-1][j]=betaipi[i][L-1][j]+ES[k][i][L-1]*
                        squeezeM[k][j];
      }
   }
}

// induction
for (int ll = L-2; ll>=0; ll--){
   for (int ii = 0; ii<N; ii++){
      for (int i=0; i<nstates; i++){
         tempRep[i]= x[ll][ii];
         tempMU[i] = MU[i][ll];
         tempSI[i] = SI[i][ll];
      }
      tempG = gaussian(tempRep, tempMU, tempSI);

      c[ll][ii]=0; //clearing the matrix.
      for (int i=0; i<M; i++){
         f[i][ll][ii] = tempG[i];
         udRet.beta[i][ll][ii] = tempG[i]*betaipi[i][ll+1][ii];
         udRet.btpni[i][ll+1][ii]= tempG[i];
         c[ll][ii]=c[ll][ii]+udRet.beta[i][ll][ii];
      }
free(tempG);
      for (int i=0; i<M; i++)
         if (scl != 0)
            udRet.beta[i][ll][ii] = udRet.beta[i][ll][ii]/c[ll][ii];

   }

   for (int i=0; i<M; i++)
      for (int j=0; j<N; j++){
         betaipi[i][ll][j]=0;  //makes sure it's zero.
         for (int k=0; k<M; k++){
            betaipi[i][ll][j]= betaipi[i][ll][j]+ES[k][i][ll]*
                           udRet.beta[k][ll][j];
         }
      }
}


// "down" step

// intialization

for (int i=0; i<M; i++)
    for (int j=0; j<N; j++)
        udRet.alpha[i][0][j] = POS[i];

if (scl != 0){
   for (int ii=0; ii<N; ii++)
      for (int i=0; i<M; i++)
         udRet.alpha[i][0][ii] = udRet.alpha[i][0][ii]/c[0][ii];
}

// induction
double *abM;
abM=createM1(M);

for (int ll = 1; ll<L; ll++){
   for (int ii = 0; ii<N; ii++){
      for (int i=0; i<M; i++)
         abM[i]=udRet.alpha[i][ll-1][ii]*udRet.btpni[i][ll][ii];
      for (int i=0; i<M; i++){
         udRet.alpha[i][ll][ii]=0;
         for (int j=0; j<M; j++)
            udRet.alpha[i][ll][ii] = udRet.alpha[i][ll][ii]+ES[i][j][ll]*abM[j];
      }
      if (scl != 0){
         for (int i=0; i<M; i++)
            udRet.alpha[i][ll][ii] = udRet.alpha[i][ll][ii]/c[ll][ii];
      }
   }
}

// calculate posterior distribution on the states
for (int i=0; i<M; i++)
   for (int j=0; j<L; j++)
      for (int k=0; k<N; k++)
         udRet.gamma[i][j][k] = udRet.alpha[i][j][k]*udRet.beta[i][j][k];

double **gammasum ;
gammasum=createM2(L,N);
for (int j=0; j<L; j++)
   for (int k=0; k<N; k++)
      for (int i=0; i<M; i++)
         gammasum[j][k]=gammasum[j][k]+udRet.gamma[i][j][k];

for (int i=0; i<M; i++)
   for (int j=0; j<L; j++)
      for (int k=0; k<N; k++)
         udRet.gamma[i][j][k] = udRet.gamma[i][j][k]/gammasum[j][k];

double *sum, *lk;
sum=createM1(N);
lk=createM1(N);

if (scl != 0){ //c is l by n
   for (int j=0; j<N; j++){
      sum[j]=0;
      for (int i=0; i<L; i++){
         sum[j]=sum[j]+log10(c[i][j]);
      }
      lk[j] = sum[j] - log10(2.0);
   }
}
else{
   for (int k=0; k<N; k++){
      sum[k]=0;
      for (int i=0; i<M; i++)
         sum[k]=sum[k]+udRet.alpha[i][0][k]*udRet.beta[i][0][k];
   lk=sum;
   }
}

free3d(betaipi,M,L); free2d(c,L); free3d(f,M,L);
free(tempRep); free(tempMU);
free(tempSI);
free2d(squeezeM,M); free(abM);
free2d(gammasum, L);
free(sum); free(lk);

return udRet;
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

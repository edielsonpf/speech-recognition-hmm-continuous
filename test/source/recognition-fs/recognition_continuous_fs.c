/*
********************************************************************************
*
	Name: recognition_continuous_fs.c

	Author: Jose

	Date:   january/96

	Description: This program recognizes an isolated word, using forward 
algorithm and continuous models. This algorithm is described in the paper " A Tutorial on Hidden Markov Models and selected Applications on Speech Recognition " of L. R. Rabiner ( february of 1989). It is considered a final state.

	Inputs: number of models, name of model files, weighting coefficients,
name of files with parameters of word to be recognized, name of files with 
spoken words and name of output file. 

	Outputs: file with performance of recognizer.	

*                                                                              * ********************************************************************************
*/



#include <math.h>           
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/times.h>

#define MAX_COEF_NUMBER 		16       /* maximum number of coefficients per vector */
#define MAX_STATES_NUMBER 		15     /* maximum number of states */
#define MAX_PARAMETERS_NUMBER 	6  /* maximum number of parameters */
#define MAX_MIXTURE_NUMBER 		5    /* maximum number of mixtures */
#define MAX_TIME 				500             /* maximum time */
#define MAX_WORDS_NUMBER 		50      /* maximum number of words */
#define MAX_MODELS_NUMBER 		1      /* maximum number of models for each word */
#define MAX_WORD_STRING_SIZE	50
#define MAX_STRING_SIZE			100
#define NUMBER_WORDS			13

  
/* global variables */

FILE   *f_in[MAX_MODELS_NUMBER][MAX_PARAMETERS_NUMBER],
       *f_in1, *f_in2,           /* input file pointers */
       *f_out;                   /* output file pointers */
 
struct mixture
{   /* mixture */
       double mean[MAX_COEF_NUMBER];       /* mixture mean vector */
       double cov_matrix[MAX_COEF_NUMBER]; /* mixture diagonal covariance matrix */
       double det;                         /* covariance matrix determinant */
};    /* end of struct mixture */

struct state
{
       double mix_coef[MAX_MIXTURE_NUMBER];    /* mixture coefficients */
       struct mixture mix[MAX_MIXTURE_NUMBER]; /* mixtures */
};   /* end of structure state */



 
/*  functions */

FILE *opening_file_read( );      /* open file for reading */
FILE *opening_file_write( );     /* open file for writing */
void reading_model( );           /* read model */
int reading_coef( );             /* read coefficients */
int reading_coef_number( );      /* read number of coefficients */
void calc_alpha( );              /* calculate forward variable - alpha */
double calc_probability( );      /* calculate probability */
void calc_symbol_probab( );      /* calculate output symbol probability density */
double calc_gaus( );             /* calculate gaussian probability density of each mixture */
void sorting_probab( );          /* sort probabilities */
void writing_header( );          /* write header */
void writing_word( );            /* write word */
void writing_result_word( );     /* write results */
void writing_result( );          /* write results */
void writing_total_result( );    /* write results */

/* main  program */
int main(int argc, char *argv[])
{

	double
	symbol_probab[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER][MAX_TIME],
	/* symbol probabilities */

	alpha[MAX_STATES_NUMBER][MAX_TIME], /* forward variable */

	scaling_factor[MAX_TIME]; /* scaling factor */

	double probab[MAX_WORDS_NUMBER];   /*  array of probability  of each word */

	double coef_vector[MAX_COEF_NUMBER]; /* vector of coefficients */

	double coef_model[MAX_MODELS_NUMBER];   /* weighting model coefficients */

	int coef_number[MAX_COEF_NUMBER]; /* number of coefficients per vector */

	int pi[MAX_STATES_NUMBER];       /* initial state probability */

	int index[MAX_WORDS_NUMBER];     /* index of words in the vocabulary */

	int wrong_word[MAX_WORDS_NUMBER];/* words wrongly recognized */

	char *word[MAX_WORDS_NUMBER];    /* words in the vocabulary */

	char data_file[MAX_PARAMETERS_NUMBER][MAX_STRING_SIZE]; /* name of file with parameters */

	char   model_name[MAX_STRING_SIZE],          /* name of  model */
		   spoken_word[MAX_WORD_STRING_SIZE],          /* spoken word */
		   last_word[MAX_WORD_STRING_SIZE],            /* last spoken word */
		   output_file[MAX_STRING_SIZE],         /* output file name */
		   word_file[MAX_STRING_SIZE];           /* file with the spoken words */

	char   date_time_f[25];          /* variable to store formatted time and date */

	struct model
	{                   /* structure to represent a model */

		   double                    /* transition probabilities  */
		   transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER];

		   struct state state_mix[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER]; /* array with mixture parameters (considering all states and all kind of parameters ) */

		   int param_number,         /* number of parameters  to create the model */
			   states_number,        /* number of states in the HMM */
			   mixture_number[MAX_PARAMETERS_NUMBER];  /* number of mixtures */

		   char word[MAX_WORD_STRING_SIZE];            /*  word that is represented by the model */

		   struct model *next;       /* pointer to new model */
	};     /* end of structure model */

	double  aux,                     /* auxiliar variable */
			old_aux,                 /* old auxiliar variable */
			cpu_time,                /* cpu time */
			sum_cpu_time = 0.0;      /* total cpu time */

	int word_number = 0;             /* number of words */

	int models_number;               /* number of models */

	int obs_time;                    /* time (number of frames in each observation sequence)*/

	int correct = 0,                 /* number of  correct recognized words */
		error = 0,                   /* number of erros */
		second = 0,                  /* number of times that a word was the second candidate */
		sum_correct = 0,             /* number of correct recognized words, considering all the words */
		sum_error = 0,               /* number of erros, considering all the words */
		sum_second = 0,              /* number of times that the words were the second candidate */
		word_frames = 0,             /* average number of frames per word */
		total_frames = 0;            /* average number of frames considering all the words */


	struct model *model_ptr = NULL,  /* pointer to struct model */
				 *first_model[MAX_MODELS_NUMBER],  /* first model in the list of models */
				 *aux_ptr;           /* auxiliar pointer to struct model */

	struct tms aux_time;             /* auxiliar variable to compute cpu time */

	time_t time_date;                /* variable to get time and date */

	register int i,j,k,l;            /* variables to control loops */

	time(&time_date);                /* getting the time and date */

	strftime(date_time_f,100,"%d-%h-%Y %X",localtime(&time_date));    /* formatting the date (dd-mm-yyyy) and time (hh:mm:ss) */

	/* testing the number of arguments in the command line */
	if (argc  < 7)
	{
		puts("Usage: recognition_continuous_fs models_number  model1 ... modelN coef_model1 ... coef_modelN input_file1 ... input_fileM  word_file output_file");
		puts("models_number: number of model");
		puts("model1: name of file with the name of model 1");
		puts("modelN: name of file with the name of model N");
		puts("coef_model1: weighting coefficient of model 1");
		puts("coef_modelN: weighting coefficient of model N");
		puts("input_file1: name of file with name of files with parameters 1 ");
		puts("input_fileM: name of file with name of files with parameters M ");
		puts("word_file: name of file with the spoken words ");
		puts("output_file: name of output file ");
		exit(1);
	}


	models_number = atoi(argv[1]);     /* number of models per word, using different parameters */

	for (i=0;i<models_number;i++)
		 coef_model[i] = atof(argv[models_number+2+i]);   /* weighting coefficients to combine the models */

	strncpy(output_file,argv[argc-1],MAX_STRING_SIZE);  /* file where the results will be written */
	strncpy(word_file,argv[argc-2],MAX_STRING_SIZE);    /* file with the spoken words */

	for (i=0;i<models_number;i++)
	{
		 f_in1 = opening_file_read(argv[2+i],"rb");  /* opening model file */
		 first_model[i] = NULL; /* inicializing pointes to structure model - these pointers point to the first pointer in  the structures sequence where are stored all the models */
		 word_number = 0;  /* setting  word counter (number of words in the vocabulary )  to zero */

		 printf("\r\nLoading Models\r\n");
		 /* reading the models of words and storing them in a structure sequence */
		 while (fscanf(f_in1,"%s",model_name) != EOF)
		 {
				if (ferror(f_in1))
				{
					printf("reading error on file %s \n",argv[2+i]);
					exit(1);
				}
				printf("Model: %s\r\n",model_name);
				/* allocating space for structure model */
				model_ptr = ( struct model * ) malloc(sizeof(struct model));
				if (model_ptr == NULL)
				{
					puts("error on allocating memory. \n");
					exit(1);
				}

				/* reading  model */
				reading_model(model_name,&(model_ptr->param_number),
						&(model_ptr->states_number),model_ptr->mixture_number,coef_number,
						model_ptr->transition_probab,model_ptr->state_mix,model_ptr->word);

				/* counting the number of words in the vocabulary */
				word[word_number++] = model_ptr->word;
				model_ptr->next = NULL;

				if (first_model[i] == NULL)
					first_model[i] = model_ptr;
				else
					aux_ptr->next = model_ptr;

				aux_ptr = model_ptr;

		 }  /* end of reading a model per word */

		 fclose(f_in1);

	}   /* end of reading all different models per word */

	/* setting initial states probability */
	/* as the left-right model is being used, the initial state is always the state 0 */
	pi[0] = 1;
	for (i=1;i<MAX_STATES_NUMBER;i++)
		 pi[i] = 0;

	k=0;
	/* opening files with the names of files with  parameters of words to be recognized */
	for (j=0;j<models_number;j++)
	{
		 for (i=0;i<first_model[j]->param_number;i++)
		 {
			  f_in[j][i] = opening_file_read(argv[2+2*models_number+k],"r");
			  k++;
		 }
	}

	f_in1= opening_file_read(word_file,"r");

	f_out = opening_file_write(output_file,"w");

	writing_header(models_number,coef_model,argv,date_time_f);

	strcpy(last_word," ");  /* last word recognized */

	for (i=0;i<word_number;i++)
		 wrong_word[i] = 0;


	/* computing cpu time */
	times(&aux_time);       /* getting cpu time */
	aux = (aux_time.tms_utime)/60.0;
	old_aux =  aux;

	printf("\r\nStarting Tests\r\n");

	while (fscanf(f_in1,"%s",spoken_word) != EOF )
	{
		/* testing the end of file of testing sequence */
		if (ferror(f_in1))
		{
		   printf("reading error on file %s \n",word_file);
		   exit(1);
		}

		printf("\r\nSpoken word: %s",spoken_word);
		for (i=0;i<word_number;i++)
			probab[i] = 0.0;  /* setting probabilities of each word in the vocabulary to zero  */

		   if (strncmp(last_word,spoken_word,MAX_WORD_STRING_SIZE) != 0 )
		   {
			   if(strcmp(last_word," ") != 0)
			   {
				  /* computing cpu time */
				  times(&aux_time);       /* getting cpu time */
				  aux = (aux_time.tms_utime)/60.0;
				  cpu_time = aux - old_aux;
				  old_aux = aux;
				  sum_cpu_time += cpu_time;

				  writing_result_word(correct,error,second,word_number,last_word,
						  wrong_word,word,cpu_time,word_frames);
				  sum_correct += correct;
				  sum_error += error;
				  sum_second += second;
				  total_frames += word_frames;
				  word_frames = 0;
				  correct = 0;
				  error = 0;
				  second = 0;
				  for (i=0;i<word_number;i++)
					   wrong_word[i] = 0;
			   }
			   writing_word(spoken_word);
			}

			l=0;

			/* calculating the probability of word to be the spoken word */
			for (j=0;j<models_number;j++)
			{
				 model_ptr = first_model[j];

				 /* reading name of files with parameters */
				 for (i=0;i<first_model[j]->param_number;i++,l++)
				 {
					  fscanf(f_in[j][i],"%s",data_file[i]);
					  if (ferror(f_in[j][i]))
					  {
						  printf("reading error on file %s \n",argv[2+2*models_number+(l)]);
						  exit(1);
					  }
				 }

				 for (k=0;k<word_number;k++)
				 {
					 printf("\r\nWord number:%d - %d\r\n",word_number,k+1);
					 /*  reading the parameters of word to be recognized */
					 for (i=0;i<first_model[j]->param_number;i++)
					 {
						 printf("\r\nParameter:%s\r\n",data_file[i]);

						 f_in2 = opening_file_read(data_file[i],"rb");
						 obs_time = 0;
						 coef_number[i] = reading_coef_number(f_in2,data_file[i]);
						 while (reading_coef(f_in2,data_file[i],coef_number[i],coef_vector) != 0)
						 {
							 calc_symbol_probab(model_ptr->states_number,
									 model_ptr->mixture_number[i],coef_number[i],coef_vector,model_ptr->state_mix[i],
									 symbol_probab[i],obs_time);    /* computing symbol probabilities */
							  obs_time++;
						 }   /* end of while */
						 fclose(f_in2);  /* closing data file */
					  } /* end of reading word to be recognized */

					 printf("\r\nCalculating Alpha");
					 calc_alpha(model_ptr->states_number,obs_time,
							  model_ptr->param_number,alpha,scaling_factor,model_ptr->transition_probab,
							  symbol_probab,pi);
					  probab[k] += coef_model[j]*calc_probability(obs_time,
							  scaling_factor,alpha[model_ptr->states_number-1][obs_time-1]);
					  model_ptr = model_ptr->next;
				 }  /* end of for (number of words ) */
			} /* end of for (number of models) - end of probabilities calculation */

			word_frames += obs_time;

			sorting_probab(probab,index,word_number);   /* sorting the probabilities to find the highest */

			printf("\r\nWriting result\r\n");
			writing_result(word,probab,index);

			/* computing number of correct and wrong recognized words */
			if (strncmp(spoken_word,word[index[0]],MAX_WORD_STRING_SIZE) == 0)
				correct++;
			else
			{
				  error++;
				  wrong_word[index[0]]++;
				  if (strncmp(spoken_word,word[index[1]],MAX_WORD_STRING_SIZE) == 0)
					  second++;
			}
			strncpy(last_word,spoken_word,MAX_WORD_STRING_SIZE);   /*  storing last spoken word */
	}    /* end of test sequence */

	printf("\r\nEnding Tests\r\n");
	/* computing cpu time */
	times(&aux_time);       /* getting cpu time */
	aux = (aux_time.tms_utime)/60.0;
	cpu_time = aux - old_aux;
	sum_cpu_time += cpu_time;

	/* writing results */
	writing_result_word(correct,error,second,models_number,last_word,wrong_word,
	word,cpu_time,word_frames);
	sum_correct += correct;
	sum_error += error;
	sum_second += second;
	total_frames += word_frames;

	/* closing data files */
	for (j=0;j<models_number;j++)
		 for (i=0;i<first_model[j]->param_number;i++)
			  fclose(f_in[j][i]);

	writing_total_result(sum_correct,sum_error,sum_second,sum_cpu_time,total_frames);

	fclose(f_out);

	for (j=0;j<models_number;j++)
	{
		 model_ptr = first_model[j];
		 while (model_ptr != NULL)
		 {
			 aux_ptr = model_ptr->next;
			 free(model_ptr);   /* memory free */
			 model_ptr = aux_ptr;
		 }

	}
	return 0;
}   /* end of main */






/*
********************************************************************************
*                                                                              *
*   opening_file_read:      				                       *
*                                                                              *
*   This function opens a file for reading.                                    *
*									       *
*   input: file_name, mode.						       *
*									       *
*   output: f_in.       						       *
*									       *
********************************************************************************
*/
 
FILE *opening_file_read(file_name,mode)
char *file_name;   /* pointer to file name */
char *mode;        /* pointer to reading mode */

{
FILE *f_in;        /* pointer to reading file */

if ((f_in = fopen(file_name,mode)) == NULL){
     printf("file %s not found \n",file_name);
     exit(1);
}

return(f_in);

}      /* end of function opening_file_read */





/*
********************************************************************************
*                                                                              *
*   opening_file_write:      				                       *
*                                                                              *
*   This function opens the file for writing.                                  *
*									       *
*   input: file_name, mode.						       *
*									       *
*   output: f_out.       						       *
*									       *
********************************************************************************
*/
 
FILE *opening_file_write(file_name,mode)
char *file_name;   /* pointer to file name */
char *mode;        /* pointer to writing mode */

{
FILE *f_out;       /* pointer to writing file */

if((f_out=fopen(file_name, mode)) == NULL) {
   printf("can't open file %s \n",file_name);
   exit(1);
}

return(f_out);

}  /* end of function opening_file_write */





/*
********************************************************************************
*                                                                              *
*   reading_coef:          				                       *
*                                                                              *
*   This function reads a vector of coefficients.                              *
*									       *
*   input: f_in, file_name, coef_number.				       *
*									       *
*   output: coef_vector, aux.    					       *
*									       *
********************************************************************************
*/
 

int reading_coef(f_in,file_name,coef_number,coef_vector)
FILE *f_in;           /* pointer to input file */
char *file_name;      /* pointer to input file name */
int coef_number;      /* number of coefficients */
double *coef_vector;  /* pointer to vector of coefficients */

{
int aux;              /* number of bytes read */

aux = fread(coef_vector,sizeof(double),coef_number,f_in); /* reading coefficients */
if (ferror(f_in)) {
    printf("reading error on file %s \n",file_name);
    exit(1);
}

return(aux);

}  /* end of function reading_coef */






/*
********************************************************************************
*                                                                              *
*   reading_coef_number:          				               *
*                                                                              *
*   This function reads the number of coefficients per frame.                  *
*									       *
*   input: f_in, file_name.		                 		       *
*									       *
*   output: coef_number.         					       *
*									       *
********************************************************************************
*/
 

int reading_coef_number(f_in,file_name)
FILE *f_in;           /* pointer to input file */
char *file_name;      /* pointer to input file name */

{
int coef_number;      /* number of coefficients */

fread(&coef_number,sizeof(int),1,f_in); /* reading number of coefficients */
if (ferror(f_in)) {
    printf("reading error on file %s \n",file_name);
    exit(1);
}

return(coef_number);

}  /* end of function reading_coef */






/*
********************************************************************************
*                                                                              *
*   reading_model:                       		                       *
*                                                                              *
*   This function reads a model.                                               *
*									       *
*   input: model_name.                                                         *
*									       *
*   output: param_number, states_number, mixture_number, coef_number,          *
*           transition_probab, state_mix, word.                                *
*									       *
********************************************************************************
*/


void reading_model(model_name,param_number,states_number,mixture_number,
coef_number,transition_probab,state_mix,word)
char *model_name;       /* name of model */
int *param_number;      /* number of parameters */
int *states_number;     /* number of states */
int mixture_number[];   /* number of mixtures */
int coef_number[];      /* number of coefficients */
double transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER];  /* transition probabilities */
struct state state_mix[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER]; /* mixture parameters */
char *word;             /* word */

{
	FILE *f_in;             /* input file */
	size_t length;          /* length of word */
	register int i,j,k;

	/* opening  model file */
	f_in = opening_file_read(model_name,"rb");

	fread(&length, sizeof(size_t),1,f_in);   /* reading length of word */
	if (ferror(f_in))
	{
		printf(" reading error on file %s",model_name);
		exit(1);
	}

	fread(word, sizeof(char),length,f_in);   /* reading word */
	if (ferror(f_in))
	{
		printf(" reading error on file %s",model_name);
		exit(1);
	}

	fread(states_number, sizeof(int),1,f_in);   /* reading number of states */
	if (ferror(f_in))
	{
		printf(" reading error on file %s",model_name);
		exit(1);
	}


	fread(param_number, sizeof(int),1,f_in);   /* reading number of parameters */
	if (ferror(f_in))
	{
		printf(" reading error on file %s",model_name);
		exit(1);
	}


	fread(mixture_number,sizeof(int),*param_number,f_in);   /* reading number of mixtures */
	if (ferror(f_in))
	{
		printf(" reading error on file %s",model_name);
		exit(1);
	}

	fread(coef_number,sizeof(int),*param_number,f_in);   /* reading number of coefficients */
	if (ferror(f_in))
	{
		printf(" reading error on file %s",model_name);
		exit(1);
	}

	for (i=0;i<*states_number;i++)
	{
		 fread(transition_probab[i], sizeof(double),*states_number,f_in);   /* reading initial model - transition probabilities */
		 if (ferror(f_in))
		 {
			 printf(" reading error on file %s",model_name);
			 exit(1);
		 }
	}

	/* reading mixture parameters */
	for (i=0;i<*param_number;i++)
	{
		 for (j=0;j<*states_number;j++)
		 {

			  /* reading mixture coefficients */
			  fread(state_mix[i][j].mix_coef,sizeof(double),mixture_number[i],f_in);
			  if (ferror(f_in))
			  {
				  printf(" reading error on file %s",model_name);
				  exit(1);
			  }

			  for (k=0;k<mixture_number[i];k++)
			  {
				   /* reading mixture mean */
				   fread(state_mix[i][j].mix[k].mean,sizeof(double),coef_number[i],	f_in);
				   if (ferror(f_in))
				   {
					   printf(" reading error on file %s",model_name);
					   exit(1);
				   }

				   /* reading covariance matrix determinant */
				   fread(&(state_mix[i][j].mix[k].det),sizeof(double),1,f_in);
				   if (ferror(f_in))
				   {
					   printf(" reading error on file %s",model_name);
					   exit(1);
				   }

				   /* reading  inverse covariance matrix (diagonal) */
				   fread(state_mix[i][j].mix[k].cov_matrix,sizeof(double),
						   coef_number[i],f_in);
				   if (ferror(f_in))
				   {
					   printf(" reading error on file %s",model_name);
					   exit(1);
				   }
			  }
		 }
	}

	fclose(f_in);    /* closing  initial model file */


}    /* end of function reading_model */








/*
********************************************************************************
*                                                                              *
*   calc_alpha:                          		                       *
*                                                                              *
*   This function calculates the forward variable (alpha).                     *
*									       *
*   input: states_number,obs_time, param_number, transition_probab,            *
*          symbol_probab,pi.                                                   *
*									       *
*   output: alpha, scaling_factor.                                             *
*									       *
********************************************************************************
*/

void calc_alpha(states_number,obs_time,param_number,alpha,
scaling_factor,transition_probab,symbol_probab,pi)
int states_number;       /* number of states */
int obs_time;            /* number of symbols */
int param_number;        /* number of parameters */
double alpha[MAX_STATES_NUMBER][MAX_TIME];  /* forward variable - alpha */
double scaling_factor[MAX_TIME];   /* scaling factor */
double transition_probab[MAX_STATES_NUMBER][MAX_STATES_NUMBER];  /* transition probabilities */
double symbol_probab[MAX_PARAMETERS_NUMBER][MAX_STATES_NUMBER][MAX_TIME];
int pi[MAX_STATES_NUMBER];    /* output symbol probabilities */
{

	register int i,j,k;
//	int symbol;             /* auxiliar variables */
	double product,
		   sum,
		   aux;
	/* computing forward variable alpha */

	/* inicialization */
	sum = 0.0;
	for (i=0;i<states_number;i++)
	{
		 product = 1.0;
		 for (j=0;j<param_number;j++)
		 {
			  product *=  symbol_probab[j][i][0];
		 }
		 alpha[i][0] = pi[i]*product;
		 sum += alpha[i][0];
	}

	scaling_factor[0] = 1.0/sum; /* scaling factor */

	for (i=0;i<states_number;i++)  /* scaling alpha[i][0] */
		 alpha[i][0] *= scaling_factor[0];

	/* induction */
	for (k=1;k<obs_time;k++)
	{
		 sum = 0.0;
		 for (i=0;i<states_number;i++)
		 {
			  aux = 0.0;
			  for (j=0;j<states_number;j++)
				   aux += alpha[j][k-1]*transition_probab[j][i];
			  product = 1.0;
			  for (j=0;j<param_number;j++)
			  {
				   product *= symbol_probab[j][i][k];
			  }
			  alpha[i][k] = aux*product;
			  sum += alpha[i][k];
		 }
		 scaling_factor[k] = 1.0/sum;

		 for (i=0;i<states_number;i++)  /* scaling alpha[i][k] */
			  alpha[i][k] *= scaling_factor[k];

	}
}   /* end of calc_alpha */






/*
********************************************************************************
*                                                                              *
*   calc_probability:                   		                       *
*                                                                              *
*   This function calculates the probability.                                  *
*									       *
*   input: obs_time, scaling_factor, alpha.                                    *
*									       *
*   output: probability.                                                       *
*									       *
********************************************************************************
*/

double calc_probability(obs_time,scaling_factor,alpha)
int obs_time;            /* time */
double *scaling_factor;  /* scaling factor */
double alpha;            /* alpha[states_number-1][obs_time-1] */

{
double probability = 0.0;
register int i;

for (i=0;i<obs_time;i++)
     probability -= log(scaling_factor[i]);

probability += log(alpha);

return(probability);

}  /* end of calc_probability */








/*
********************************************************************************
*                                                                              *
*   calc_symbol_probab:                   		                       *
*                                                                              *
*   This function calculates the output symbol density probabilities.          *
*									       *
*   input: states_number, mixture_number,coef_number, coef_vector, state_mix,  *
*          obs_time.                                                           *
*									       *
*   output: symbol_probab.                                                     *
*									       *
********************************************************************************
*/

void calc_symbol_probab(states_number,mixture_number,coef_number,coef_vector,
state_mix,symbol_probab,obs_time)
int states_number;    /* number of states */
int mixture_number;   /* number of mixtures */
int coef_number;      /* number of coefficients per vector */
double *coef_vector;  /* vector of coefficients */
struct state state_mix[MAX_STATES_NUMBER];  /* mixture parameters */

double symbol_probab[MAX_STATES_NUMBER][MAX_TIME];  /* gaussian probability density in each state */
int obs_time;         /* time */

{
double gauss;  /* gaussian probability density in each mixture */
register int i,j;

for (i=0;i<states_number;i++) {
     symbol_probab[i][obs_time] = 0.0;
     for (j=0;j<mixture_number;j++) {
          gauss = calc_gaus(coef_number,coef_vector,
state_mix[i].mix[j].mean,state_mix[i].mix[j].cov_matrix,
state_mix[i].mix[j].det);
          gauss *= state_mix[i].mix_coef[j];
          symbol_probab[i][obs_time] += gauss;       
     } 

     
}   


}   /* end of function calc_symbol_probab */





/*
********************************************************************************
*                                                                              *
*   calc_gaus:                           		                       *
*                                                                              *
*   This function calculates the gaussian probability density using diagonal   *
*   covariance matrix.                                                         *
*									       *
*   input: coef_number, coef_vector, mean, inv_cov, det.                       *
*									       *
*   output: gaus.                                                              *
*									       *
********************************************************************************
*/

double calc_gaus(coef_number,coef_vector,mean,inv_cov,det)
int coef_number;                     /* number of coefficients per vector */
double coef_vector[MAX_COEF_NUMBER]; /* vector of coefficients */
double mean[MAX_COEF_NUMBER];        /* mixture mean vector */
double inv_cov[MAX_COEF_NUMBER];     /* inverse covariance matrix */
double det;                          /* covariance matrix determinant */
{

register int i;
double aux = 0.0,                    /* auxiliar variable */
       aux1,
       aux2;
double dif,
       gaus;

/* gaussian probability density computation */

aux1 = 2.0*M_PI;
aux2 = coef_number/2.0;
aux1 = pow(aux1,aux2);

if (det != 0) {
    aux2 = fabs(det);
    aux2 = pow(aux2,0.5);
   
    for (i=0;i<coef_number;i++) { 
         dif = coef_vector[i] - mean[i];
         aux += dif*inv_cov[i]*dif;
    }
    aux *= (-0.5);
    aux = exp(aux);
    
    gaus = aux/(aux1*aux2);
}

return(gaus);

}   /* end of function calc_gaus */






/*
********************************************************************************
*                                                                              *
*   sorting_probab:      				                       *
*                                                                              *
*   This function sorts the probabalities.                                     *
*									       *
*   input: probab, index, word_number.                            	       *
*									       *
*   output: index.                                                             *
*									       *
********************************************************************************
*/

void sorting_probab(probab,index,word_number)
double *probab;     /* probabilities of words in the vocabulary */
int *index;         /* index of words in the vocabulary */
int word_number;    /* spoken word */

{
int done = 0;
int aux;
register int i;

for (i=0;i<word_number;i++)
     index[i] = i;

while (!done){
       done = 1;
       for (i=0;i<(word_number-1);i++){
            if (probab[index[i]] < probab[index[i+1]]) {
                aux = index[i];
                index[i] = index[i+1];
	        index[i+1] = aux;
	        done = 0;
            }  /* end of if */
       }  /* end of for */
         
}  /* end of while */


}    /* end of function sorting_probab */




/*
********************************************************************************
*                                                                              *
*   writing_header:      				                       *
*                                                                              *
*   This function writes the header in the recognition resulting file.         *
*									       *
*   input: models_number, coef_model, argv, date_time_f.              	       *
*									       *
*   output: void.                                                              *
*									       *
********************************************************************************
*/

void writing_header(models_number,coef_model,argv,date_time_f)
int models_number;    /* number of model */
int *coef_model;      /* weighting model coefficients */
char *argv[];
char *date_time_f;    /* date and time */

{
	register int i;

	fprintf(f_out,"Isolated word recognition using Continuous HMM (diagonal covariance matrix). It is considered a final state. \n");
	fprintf(f_out,"Algorithm used for recognition: Forward \n");
	fprintf(f_out,"Number of models: %d  \n",models_number);
	for (i=0;i<models_number;i++)
	{
		 fprintf(f_out,"Model name %d: %s\n",(i+1),argv[2+i]);
		 fprintf(f_out,"Weighting coefficient of model %d:%.2d\n",(i+1),coef_model[i]);
	}
	fprintf(f_out,"Date and time: %s \n\n",date_time_f);
}     /* end of function writing_header */




/*
********************************************************************************
*                                                                              *
*   writing_word:        				                       *
*                                                                              *
*   This function writes the spoken word.                                      *
*									       *
*   input: word.                                                 	       *
*									       *
*   output: void.                                                              *
*									       *
********************************************************************************
*/

void writing_word(word)
char *word;   /* spoken word */

{

fprintf(f_out,"\nSpoken word: %s\n",word);

}   /* end of function writing_word */




/*
********************************************************************************
*                                                                              *
*   writing_result:      				                       *
*                                                                              *
*   This function writes the recognition results.                              *
*									       *
*   input: word, probab, index.                                  	       *
*									       *
*   output: void.                                                              *
*									       *
********************************************************************************
*/

void writing_result(word,probab,index)
char *word[MAX_WORDS_NUMBER];  /* words in the vocabulary */
double *probab;                /* probability of words in the vocabulary */
int *index;                    /* index of words in the vocabulary */
{
	register int i;

	for(i=0;i<NUMBER_WORDS;i++)
	{
		 printf("%s :  %f \n",word[index[i]],probab[index[i]]);
	}
	printf("\n");
}   /* end of function writing_result */



/*
********************************************************************************
*                                                                              *
*   writing_result_word:      				                       *
*                                                                              *
*   This function writes the recognition results.                              *
*									       *
*   input: correct, error, second, word_number, spoken_word, wrong_word, word, *
*          cpu_time, word_frames.                                              *
*									       *
*   output: void.                                                              *
*									       *
********************************************************************************
*/


void writing_result_word(correct,error,second,word_number,spoken_word,
wrong_word,word,cpu_time,word_frames)
int correct;         /* number of words correctly recognized */
int error;           /* number of errors */
int second;          /* number of times that the correct word is the second candidate */
int word_number;     /* number of words in the vocabulary */
char *spoken_word;   /* spoken word */
int *wrong_word;     /* words wrongly recognized */
char *word[MAX_WORDS_NUMBER];  /* words in the vocabulary */
double cpu_time;     /* cpu time */
int word_frames;     /* average number of frames */

{
double per; 
int sum;
register int i;

sum = correct + error;
per = (double) correct/ (double) sum;
cpu_time /= sum;
word_frames /= sum;

fprintf(f_out,"\nResults: \n");
fprintf(f_out,"Spoken word: %s\n",spoken_word);
fprintf(f_out,"Correct words: %d\n",correct);
fprintf(f_out,"Errors: %d\n",error);
fprintf(f_out,"Percentagen correct : %.2f%%\n",(per*100.0));
fprintf(f_out,"Second candidate: %d\n",second);
if (error != 0) {
    fprintf(f_out,"Wrong words: \n");
    for (i=0;i<word_number;i++){
         if (wrong_word[i] != 0)
             fprintf(f_out,"%s: %d time%s\n",word[i],wrong_word[i],
wrong_word[i]==1 ? "" : "s");
            }

}
fprintf(f_out,"Average recognition time: %.2f sec. \n",cpu_time);
fprintf(f_out,"Average word length: %d frames \n",word_frames);

}  /* end of function writing_result_word */





/*
********************************************************************************
*                                                                              *
*   writing_total_result:      				                       *
*                                                                              *
*   This function writes the recognition results.                              *
*									       *
*   input: correct, error, second, cpu_time, total_frames.           	       *
*									       *
*   output: void.                                                              *
*									       *
********************************************************************************
*/

void writing_total_result(correct,error,second,cpu_time,total_frames)
int correct;       /* number of words correctly recognized */
int error;         /* number of erros */
int second;        /* number of times that the correct word is the second candidate */
double cpu_time;   /* cpu time */
int total_frames;  /* average number of frames */

{
double per; 
int sum;

sum = correct + error;
per = (double) correct/ (double) sum;
cpu_time /= sum;
total_frames /= sum;

fprintf(f_out,"\nConsidering all the words: \n");
fprintf(f_out,"Results: \n");
fprintf(f_out,"Correct words: %d\n",correct);
fprintf(f_out,"Errors: %d\n",error);
fprintf(f_out,"Percentagen correct : %.2f%%\n",(per*100.0));
fprintf(f_out,"Second candidate: %d\n",second);
fprintf(f_out,"Average recognition time: %.2f sec. \n",cpu_time);
fprintf(f_out,"Average word length: %d frames \n",total_frames);

}     /* end of function writing_total_result */





 

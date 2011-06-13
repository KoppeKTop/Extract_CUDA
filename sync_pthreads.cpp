/*
 * threads_test.c
 *
 *  Created on: 06.03.2009
 *      Author: andrey
 */

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

int _RUNNING_TH = 4;

class th_ctrl {
public:
	th_ctrl(unsigned int);
	~th_ctrl();
	unsigned int ended_cnt;
	pthread_mutex_t * lock;
	sem_t * sems;
	inline int getCtrlCnt() {	return ctrl_cnt;	}
	void setCtrlCnt(unsigned int);
private:
	unsigned int ctrl_cnt;
};

th_ctrl::th_ctrl(unsigned int N):ended_cnt(0),ctrl_cnt(N)
{
	// printf("Create sync struct for %i threads\n", ctrl_cnt);
	lock = new pthread_mutex_t;
	if (pthread_mutex_init(lock, NULL) != 0) {
		printf("Error while creating th_ctrl");
	}
	if (N <= 1) 
	{
		sems = new sem_t[1];
		sem_init(&sems[0], 0, 0);
		return;
	}
	sems = new sem_t[N-1];
	int i;
	for (i = 0; i < N-1; i++)
		sem_init(&sems[i], 0, 0);
}

th_ctrl::~th_ctrl()
{
	pthread_mutex_lock(this->lock);
	delete [] this->sems;
	pthread_mutex_unlock(this->lock);
	delete this->lock;
}

void th_ctrl::setCtrlCnt(unsigned int N)
{
	pthread_mutex_lock(this->lock);
	if (N <= 1) 
	{
		delete [] this->sems;
		sems = new sem_t[1];
		sem_init(&sems[0], 0, 0);
		ctrl_cnt = N;
	} else {
		if (N != ctrl_cnt)
		{
			delete [] this->sems;
			sems = new sem_t[N-1];
			ctrl_cnt = N;
		}
		int i;
		for (i = 0; i < N-1; i++)
			sem_init(&sems[i], 0, 0);
	}
	pthread_mutex_unlock(this->lock);
}

th_ctrl *th;

extern "C" void sync_pthreads()
{
	int i;
	if (th == NULL)
	{
		printf("ERROR: Thread control not initialized. Use init_sync(N)\n");
		return;
	}
	if (th->getCtrlCnt() <= 1) return;

	pthread_mutex_lock(th->lock);
	if (++th->ended_cnt == th->getCtrlCnt()) {
		//sem_init(&th->sems[0], 0, th->ended_cnt-1);
		for(i = 0; i < th->getCtrlCnt()-1; i++)
		{
			sem_post(&th->sems[i]);
		}
		th->ended_cnt = 0;
		pthread_mutex_unlock(th->lock);
		return;
	}
	i = th->ended_cnt-1;
	pthread_mutex_unlock(th->lock);
	sem_wait(&th->sems[i]);
}

extern "C" void init_sync(int N)
{
	if (th == NULL) th = new th_ctrl(N);
	else th->setCtrlCnt(N);
}

/*
void * thread_func(void *arg)
{
   int i;
   int loc_id = * (int *) arg;
   int *result = new int;

   printf("Thread %i is running\n", loc_id);
   if (loc_id == 0) {
	   printf("First thread begin to think\n");
	   for (i = 5; i > 0; i--){
		   printf("Wait for %i seconds\n", i);
		   sleep(1);
	   }
   }
   sync_pthreads();
   printf("separate line\n");
   sync_pthreads();

   for (i = 0; i < 4; i++) {
	   printf("Thread %i is running\n", loc_id);
	   sleep(1);
   }

   printf("Thread %i is ending\n", loc_id);
   *result = 0;
   return result;
}

int main(int argc, char * argv[])
{
	int ids[4], result, i;
	pthread_t threads[4];
	th = new th_ctrl(_RUNNING_TH);

	for (i = 0; i<_RUNNING_TH; i++)
	{
		ids[i] = i;
		result = pthread_create(&threads[i], NULL, thread_func, &ids[i]);
		if (result != 0) {
			  perror("Creating thread");
			  return EXIT_FAILURE;
			}
	}
	for (i = 0; i<_RUNNING_TH; i++) {
		result = pthread_join(threads[i], NULL);
		if (result != 0) {
		  perror("Joining thread");
		  return EXIT_FAILURE;
		}
	}
	printf("Done\n");
	delete th;
	return EXIT_SUCCESS;
}
*/


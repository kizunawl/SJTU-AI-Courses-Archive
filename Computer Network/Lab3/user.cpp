// Copyright by Kizunawl

/*
⣿⣿⣿⣿⣿⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠹⢿⣿⣿⣿⠀⠀⠀⠀⠀
⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⢿⣿⠀⠀⠀⠀⠀
⣿⣿⣿⡿⠛⠁⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀
⣿⣿⡟⠀⠀⠀⠱⠟⠛⠛⠒⠖⠀⠀⠀⠀⠀⠀⠀⠀⢤⣦⣴⣼⡟⠀⠀⠀⠀⠀
⣿⣿⡇⠀⠀⠀⠀⠀⠀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠇⠀⠀⠀⠀⠀
⣿⣿⠀⠀⠀⠀⠀⠀⠉⠛⠛⠀⠑⠀⠀⠀⠀⠀⠐⢾⡟⠑⠆⠀⠀⠀⠀⠀⠀⠀
⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⡃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠤⠤⡀⢠⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢻⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠸⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⡠⠤⠄⠠⢀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⡆⣿⣿⣤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⡇⠘⢿⣿⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠀⠀⠙⠿⣿⣿⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⣦⠀⠀⠀⠈⠙⠿⣿⣿⣷⣤⣄⣀⣠⣠⠐⣆⡠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
*/

#include<sys/types.h>
#include<sys/socket.h>
#include<netdb.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<unistd.h>
#include<errno.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<iostream>
#include<pthread.h>

#define PORT 1145 /* . . . -- */
#define MAXDATALEN 1024
#define SERVERNAME "10.0.0.5"


int init(){
	int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = inet_addr(SERVERNAME);

    connect(sockfd, (struct sockaddr*)&addr, sizeof(addr));

	return sockfd;

}

void* sendMsg(void* p){
	int sockfd = *(int*)p;
	printf("# --- OK to send to ip %s --- #\n", SERVERNAME);

	while (1){
		char message[1024] = {};
		std::cin.getline(message, MAXDATALEN);
		send(sockfd, message, strlen(message), 0);
	}
	close(sockfd);
}

void* rcvMsg(void* p){
	int sockfd = *(int*)p;
	printf("# --- OK to rcv from ip %s --- #\n", SERVERNAME);

	while (1){
		char buffer[1024] = {};
		int numbytes = recv(sockfd, buffer, MAXDATALEN-1, 0);
		buffer[numbytes] = '\0';
		printf("%s\n", buffer);
	}
	close(sockfd);
}

int main(int argc, char **argv){
	int sockfd = init();
	pthread_t send_pid, rcv_pid;
	pthread_create(&send_pid, 0, sendMsg, &sockfd);
	pthread_create(&rcv_pid, 0, rcvMsg, &sockfd);
	pthread_exit(NULL);
	return 0;
}
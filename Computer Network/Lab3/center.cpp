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
#include<pthread.h>

#define PORT 1145 /* . . . -- */
#define BACKLOG 64
#define MAXDATALEN 1024

int fds[10];
pthread_t pids[64];
int fd_size = 0, pid_size = 0;

int init(){
	int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    // addr.sin_addr.s_addr = inet_addr(SERVERNAME);

    bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));

	// start listen
	listen(sockfd, BACKLOG);

	return sockfd;
}

void* rcvMsg(void* p){
	int tempfd = *(int*)p;
	// int count = 0;
	struct sockaddr_in addr, rcv_addr;
	socklen_t addrlen = sizeof(addr), rcv_addrlen = sizeof(rcv_addr);
	int name = getpeername(tempfd, (struct sockaddr*)&addr, &addrlen);

	while (1){
		// printf("%d\n", ++count);
		char buffer[1024] = {};
		int numbytes = recv(tempfd, buffer, MAXDATALEN-1, 0);
		if (-1 == numbytes){
			perror("recv fail");
			exit(-1);
		}
		// if (0 == numbytes){
		// 	break;
		// }

		// char* clientIP = inet_ntoa(rcv_addr.sin_addr);
		char *clientIP = inet_ntoa(addr.sin_addr);

		// printf("%s\n", clientIP);
		buffer[numbytes] = '\0';
		char target_ip[] = "10.0.0.0";
		char format[] = " From h";
		target_ip[7] = buffer[4];
		// struct hostent* targethost;
		// targethost = gethostbyname("h2");
		// for (int i=0;targethost->h_addr_list[i];++i){
		// 	printf("%s\n", inet_ntoa(*(struct in_addr*)(targethost->h_addr_list[i])));
		// }
		// char* target_ip = inet_ntoa(*(struct in_addr*)(targethost->h_addr_list[0]));

		// printf("%d", numbytes);
		// printf("%s\n", target_ip);

		char outputbuf[1024];
		for (int i = 7; i < numbytes; ++i){
			outputbuf[i-7] = buffer[i];
		}
		for (int i = 0; i < 7; ++i){
			outputbuf[numbytes-7+i] = format[i];
		}
		for (int i = 7; i < strlen(clientIP); ++i){
			outputbuf[numbytes-7+i] = clientIP[i];
		}
		outputbuf[numbytes-7+strlen(clientIP)] = '\0';
		// printf("%s\n", outputbuf);

		for (int i = 0; i < fd_size; ++i){
			struct sockaddr_in peer_addr;
			int peer_addrlen = sizeof(peer_addr);
			int peer_name = getpeername(fds[i], (struct sockaddr*)&peer_addr, (socklen_t*)&peer_addrlen);
			if (strcmp(inet_ntoa(peer_addr.sin_addr), target_ip) == 0){
				send(fds[i], outputbuf, strlen(outputbuf), 0);
				break;
			}
		}
		// printf("%s\n", buffer);
		// if (strcmp(buffer, "exit") == 0){
		// 	break;
		// }
	}
	close(tempfd);
}

void start(int sockfd){
	printf("# --- Starting Server --- #\n");
	// printf("%d\n", sockfd);
	// int count = 0;
	while (1){
		// New addr to accept client addr
		// printf("# %d\n", ++count);
		int new_fd = accept(sockfd, NULL, NULL);
		if (-1 == new_fd){
			perror("accept client fail");
			exit(-1);
		}
		fds[fd_size++] = new_fd;

		struct sockaddr_in peer_addr;
		int peer_addrlen = sizeof(peer_addr);
		int peer_name = getpeername(new_fd, (struct sockaddr*)&peer_addr, (socklen_t*)&peer_addrlen);
		printf("# --- ip %s Connected --- #\n", inet_ntoa(peer_addr.sin_addr));

		++pid_size;
		pthread_create(&pids[pid_size-1], 0, rcvMsg, &fds[fd_size-1]);
	}
	pthread_exit(NULL);
}

int main(){
	int sockfd = init();
	start(sockfd);
	return 0;
}
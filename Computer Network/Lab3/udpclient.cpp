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
#include<net/if.h>
#include<arpa/inet.h>
#include<unistd.h>
#include<errno.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<iostream>
#include<pthread.h>
#include<sys/ioctl.h>

#define PORT 1145 /* . . . -- */
#define MAXDATALEN 1024
#define BROADCASTIP "10.255.255.255"

char myIP[20][20];
int myIP_size = 0;


void* sendMsg(void* p){
	int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    int broadcast = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(1145);
    addr.sin_addr.s_addr = inet_addr(BROADCASTIP);

    printf("# --- OK to send --- #\n");

    while (1){
        char buffer[MAXDATALEN];
        std::cin.getline(buffer, MAXDATALEN);
        sendto(sockfd, buffer, strlen(buffer), 0, (struct sockaddr*)&addr, sizeof(addr));
    }

    close(sockfd);
}

void* rcvMsg(void* p){
	int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));

    struct sockaddr_in rcv_addr;
    socklen_t rcv_addrlen = sizeof(rcv_addr);
    rcv_addr.sin_family = AF_INET;
    rcv_addr.sin_port = htons(PORT);
    rcv_addr.sin_addr.s_addr = INADDR_ANY;

    printf("# --- OK to Receive --- #\n");

    while (1){
        char buffer[1024] = {};
        recvfrom(sockfd, buffer, MAXDATALEN-1, 0, (struct sockaddr*)&rcv_addr, &rcv_addrlen);

        bool not_from_me = true;
        for (int i = 0; i < myIP_size; ++i){
            if (strcmp(inet_ntoa(rcv_addr.sin_addr), myIP[i]) == 0){
                not_from_me = false;
                break;
            }
        }
        if (true == not_from_me){
            printf("%s\n", buffer);
        }
    }
    close(sockfd);
}

void getmyIP(){
	int sockfd;
	struct ifconf ifc;
	char buffer[1024], ip[20];
	struct ifreq *ifr;
 
	ifc.ifc_len = 1024;
	ifc.ifc_buf = buffer;
 
	sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	ioctl(sockfd, SIOCGIFCONF, &ifc);
	ifr = (struct ifreq*)buffer;
 
	for(int i = (ifc.ifc_len/sizeof(struct ifreq)); i; --i){
		// printf("net name: %s\n", ifr->ifr_name);
		inet_ntop(AF_INET,&((struct sockaddr_in *)&ifr->ifr_addr)->sin_addr, ip, 20);
		// printf("ip: %s \n", ip);
        strncpy(myIP[myIP_size++], ip, 20);
		++ifr;
	}
}

int main(int argc, char **argv){
    getmyIP();
	pthread_t send_pid, rcv_pid;
	pthread_create(&send_pid, 0, sendMsg, NULL);
	pthread_create(&rcv_pid, 0, rcvMsg, NULL);
	pthread_exit(NULL);
	return 0;
}
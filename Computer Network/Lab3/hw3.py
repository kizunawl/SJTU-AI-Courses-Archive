#!/usr/bin/env python

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.node import IVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf
from subprocess import call

def myNetwork():

    net = Mininet()

    info( '*** Adding controller\n' )
    c0=net.addController('c0')

    info( '*** Add switches\n')
    s1 = net.addSwitch('s1')

    info( '*** Add hosts\n')
    host = []
    for i in range(5):
        host.append(net.addHost(f'h{i+1}'))

    info( '*** Add links\n')
    for i in range(5):
        net.addLink(s1, host[i])

    info( '*** Starting network\n')
    net.build()
    net.start()

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    myNetwork()


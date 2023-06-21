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
    s2 = net.addSwitch('s2')
    s3 = net.addSwitch('s3')

    info( '*** Add hosts\n')
    h1 = net.addHost('h1')
    h2 = net.addHost('h2')
    h3 = net.addHost('h3')
    h4 = net.addHost('h4')

    info( '*** Add links\n')
    net.addLink(h4, s1)
    net.addLink(s1, s3, bw=20, ls=5)
    net.addLink(s3, h3)
    net.addLink(s1, s2, b2=20, ls=5)
    net.addLink(s2, h2)
    net.addLink(s1, h1)

    info( '*** Starting network\n')
    net.build()
    net.start()
    # info( '*** Starting controllers\n')
    # for controller in net.controllers:
    #     controller.start()

    # info( '*** Starting switches\n')
    # net.get('s1').start([c0])
    # net.get('s2').start([c0])
    # net.get('s3').start([c0])

    # info( '*** Post configure switches and hosts\n')

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    myNetwork()


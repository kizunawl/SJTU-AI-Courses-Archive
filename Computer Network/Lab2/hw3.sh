#!/bin/sh

ovs-ofctl del-flows s1
ovs-ofctl del-flows s2
ovs-ofctl del-flows s3

ovs-ofctl add-flow s2 "in_port=3, actions=drop"
ovs-ofctl add-flow s3 "in_port=3, actions=drop"
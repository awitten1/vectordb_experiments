# The two CloudLab nodes (Wisconsin) for the Hacker News study. Sourced
# on the workstation by the hn_*.sh drivers; source it by hand before
# sync.sh too:   . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh push
#
# Roles: SERVER runs Postgres; CLIENT runs the network clients and Ginger.
# SERVER_IP is the server's address on the experiment LAN; NIC is that link's
# interface name on the server (watched for drops during measurements).
#
# Since 2026-11-24: two c220g5 nodes (2x Xeon Silver 4114, AVX-512, 192 GB).
# c220g5-110902 has 10.10.1.2 and c220g5-110917 has 10.10.1.1, on enp94s0f0np0.
# Before: the c220g2 pairs c220g2-011111 / c220g2-011114, then
# c220g2-010619 / c220g2-010620 (on enp6s0f0), whose results are in
# bench/results/<host>/.
export SERVER=c220g5-110902
export CLIENT=c220g5-110917
export DOMAIN=wisc.cloudlab.us
export SERVER_IP=10.10.1.2
export CLIENT_NET=10.10.1.0/24
export NIC=enp94s0f0np0

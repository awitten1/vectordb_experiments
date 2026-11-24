# The two CloudLab c220g2 nodes (Wisconsin) for the Hacker News study. Sourced
# on the workstation by bench/cloudlab/hn_qps.sh; source it by hand before
# sync.sh too:   . bench/cloudlab/nodes-wisc.sh && bench/cloudlab/sync.sh push
#
# Roles: SERVER runs Postgres; CLIENT runs the network clients and Ginger.
# SERVER_IP is the server's address on the experiment LAN (enp6s0f0); on this
# allocation c220g2-010619 has 10.10.1.2 and c220g2-010620 has 10.10.1.1.
# (Until 2026-11-24 the pair was c220g2-011111 / c220g2-011114; the server's
# link there kept dropping, so the study moved, data copied across as is.)
export SERVER=c220g2-010619
export CLIENT=c220g2-010620
export DOMAIN=wisc.cloudlab.us
export SERVER_IP=10.10.1.2
export CLIENT_NET=10.10.1.0/24

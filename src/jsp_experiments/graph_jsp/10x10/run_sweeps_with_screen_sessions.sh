
screen -r nasuta_ls
cd /home/an148650/jsp-reward-comparison/src/jsp_experiments/graph_jsp/10x10
chmod a+x orb04_nasuta_ls_graph_jsp_env_tuning_wrapper.sh
bash orb04_nasuta_ls_graph_jsp_env_tuning_wrapper.sh &
screen -X detach

screen -r nasuta_no_ls
cd /home/an148650/jsp-reward-comparison/src/jsp_experiments/graph_jsp/10x10
chmod a+x orb04_nasuta_no_ls_graph_jsp_env_tuning_wrapper.sh
bash orb04_nasuta_no_ls_graph_jsp_env_tuning_wrapper.sh &
screen -X detach

screen -r zhang_ls
cd /home/an148650/jsp-reward-comparison/src/jsp_experiments/graph_jsp/10x10
chmod a+x orb04_zhang_ls_graph_jsp_env_tuning_wrapper.sh
bash orb04_zhang_ls_graph_jsp_env_tuning_wrapper.sh &
screen -X detach

screen -r zhang_no_ls
cd /home/an148650/jsp-reward-comparison/src/jsp_experiments/graph_jsp/10x10
chmod a+x orb04_zhang_no_ls_graph_jsp_env_tuning_wrapper.sh
bash orb04_zhang_no_ls_graph_jsp_env_tuning_wrapper.sh &
screen -X detach

screen -r sam_ls
cd /home/an148650/jsp-reward-comparison/src/jsp_experiments/graph_jsp/10x10
chmod a+x orb04_samsonov_ls_graph_jsp_env_tuning_wrapper.sh
bash orb04_samsonov_ls_graph_jsp_env_tuning_wrapper.sh &
screen -X detach

screen -r sam_no_lstassel_ls
cd /home/an148650/jsp-reward-comparison/src/jsp_experiments/graph_jsp/10x10
chmod a+x orb04_samsonov_no_ls_graph_jsp_env_tuning_wrapper.sh
bash orb04_samsonov_no_ls_graph_jsp_env_tuning_wrapper.sh &
screen -X detach

screen -r tassel_no_ls
cd /home/an148650/jsp-reward-comparison/src/jsp_experiments/graph_jsp/10x10
chmod a+x orb04_graph-tassel_ls_graph_jsp_env_tuning_wrapper.sh
bash orb04_graph-tassel_ls_graph_jsp_env_tuning_wrapper.sh &
screen -X detach

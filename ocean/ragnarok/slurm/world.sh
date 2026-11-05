# Throwaway rAthena worlds inside a Slurm job. Source this file from a job script, then:
#   world_db_start             one MariaDB for the job, on node-local disk
#   world_create W NBOTS       world W: its own database, ports and bot accounts botW_0001...
#   world_start W              its login, char and map servers; returns once the map server is ready
#   world_stop_all             servers, then MariaDB; deletes everything under $WORLD_ROOT
# World W listens on 127.0.0.1 only: login B+10W, char B+10W+1, map B+10W+2, where B is
# WORLD_PORT_BASE (default 7000; the training job picks one per job so jobs can share a node).
# Its accounts file is $WORLD_ROOT/wW/accounts ("userid password" lines, mode 600). Passwords are
# random per job and never printed. Logs are copied to $WORLD_LOGS when the world stops.
# WORLD_MAP_CONF holds extra map-server settings, one per line (e.g. "lockstep: yes").
# WORLD_LEAN=1 loads only the maps in WORLD_MAPS (default: the start point's), their monsters and
# server/npc/*.txt (with @rlreset <task>). The monsters come from npc/pre-re or npc/re, by the rules
# $RATHENA was built with; world_check_client says whether the bot client speaks that build.

RO=/home/user02198/ragnarok
RG=${RG:-$RO/PufferLib/ocean/ragnarok}  # this env's directory: server/npc, client/
RATHENA=${RATHENA:-$RO/rathena}
WORLD_IMG=${WORLD_IMG:-$RO/images/mariadb-noble.sqsh}
WORLD_ROOT=${WORLD_ROOT:-/tmp/ro-world-$SLURM_JOB_ID}
WORLD_LOGS=${WORLD_LOGS:-$RO/train/logs}
WORLD_DB_PORT=${WORLD_DB_PORT:-23400}
WORLD_PORT_BASE=${WORLD_PORT_BASE:-7000}
# Accounts per world: NBOTS x WORLD_ACCOUNT_BLOCKS. Each env start (each sweep trial) claims the
# next block, because rAthena keeps a just-logged-out account "online" for up to 30 s.
WORLD_ACCOUNT_BLOCKS=${WORLD_ACCOUNT_BLOCKS:-1}
WORLD_DB_NAME=ro-wdb-$SLURM_JOB_ID
# The training field and what a new character starts with: Main Gauche (a level-1 dagger) in
# the right hand, a Cotton Shirt, 30 Red Potions.
WORLD_START_POINT=${WORLD_START_POINT:-prt_fild08,200,200}
WORLD_START_ITEMS=${WORLD_START_ITEMS:-1207,1,2:2301,1,16:501,30,0}
# Every training task's map, for lean training worlds; server/npc/rl_train.txt lists the same maps.
RG_TASK_MAPS="prt_fild08 pay_fild01 pay_fild02 mjolnir_01 prt_fild03"
WORLD_PIDS=()

wlog() {
    echo "$(date '+%F %T') $*"
}

# The client version and rules (pre-re or re) $RATHENA was configured with, from its Makefile.
world_packetver() {
    grep -o -- '-DPACKETVER=[0-9]*' $RATHENA/src/map/Makefile | head -1 | cut -d= -f2
}

world_rules() {
    if grep -q -- '-DPRERE' $RATHENA/src/map/Makefile; then echo pre-re; else echo re; fi
}

# Fails, saying why, when the bot client's packet table is for another version or rules than $RATHENA.
world_check_client() {
    local hdr=$RG/client/rathena_packets.h ver renewal want_renewal
    ver=$(awk '$2 == "RO_PACKETVER" {print $3}' $hdr)
    renewal=$(awk '$2 == "RO_RENEWAL" {print $3}' $hdr)
    want_renewal=$([ "$(world_rules)" = re ] && echo 1 || echo 0)
    if [ "$ver" != "$(world_packetver)" ] || [ "${renewal:-0}" != "$want_renewal" ]; then
        wlog "the bot client speaks $ver $([ "${renewal:-0}" = 1 ] && echo re || echo pre-re), but $RATHENA is built for $(world_packetver) $(world_rules): regenerate client/rathena_packets.h or set RATHENA"
        return 1
    fi
}

world_secret() {
    python3 -c 'import secrets; print(secrets.token_hex(11))'
}

wdb() {
    enroot start --mount $WORLD_ROOT/db:/var/lib/mysql --mount $WORLD_ROOT/run:/run/mysqld $WORLD_DB_NAME "$@"
}

wsql() {
    wdb mariadb --socket=/run/mysqld/mysqld.sock "$@"
}

world_db_start() {
    umask 077
    mkdir -p $WORLD_ROOT/db $WORLD_ROOT/run $WORLD_LOGS
    if ! enroot create -n $WORLD_DB_NAME $WORLD_IMG > $WORLD_ROOT/enroot.log 2>&1; then
        wlog "enroot create failed: $(tail -3 $WORLD_ROOT/enroot.log)"
        return 1
    fi
    wdb mariadb-install-db --datadir=/var/lib/mysql --user=$(id -un) --skip-test-db > $WORLD_ROOT/install.log 2>&1
    wdb mariadbd --datadir=/var/lib/mysql --socket=/run/mysqld/mysqld.sock --bind-address=127.0.0.1 \
        --port=$WORLD_DB_PORT --skip-name-resolve > $WORLD_ROOT/mariadb.log 2>&1 &
    WORLD_DB_PID=$!
    for i in $(seq 120); do
        wdb mariadb-admin --socket=/run/mysqld/mysqld.sock ping > /dev/null 2>&1 && break
        sleep 1
    done
    if ! wdb mariadb-admin --socket=/run/mysqld/mysqld.sock ping > /dev/null 2>&1; then
        wlog "mariadb did not start: $(tail -3 $WORLD_ROOT/mariadb.log)"
        return 1
    fi
    WORLD_DB_PW=$(world_secret)
    wsql <<EOF
CREATE USER 'ragnarok'@'127.0.0.1' IDENTIFIED BY '$WORLD_DB_PW';
EOF
    wlog "mariadb up on port $WORLD_DB_PORT after ${i}s"
}

# A runtime tree for world W: rAthena's binaries, db and npc linked in, conf linked file by file
# except conf/import, which holds this world's settings.
world_create() {
    local w=$1 nbots=$2
    local dir=$WORLD_ROOT/w$w lp=$((WORLD_PORT_BASE + 10 * w)) cp=$((WORLD_PORT_BASE + 10 * w + 1)) mp=$((WORLD_PORT_BASE + 10 * w + 2))
    local dbname=ro_w$w inter_pw bot_pw
    inter_pw=$(world_secret)
    bot_pw=$(world_secret)
    mkdir -p $dir/conf/import $dir/log
    for f in login-server char-server map-server db sql-files; do
        ln -s $RATHENA/$f $dir/$f
    done
    for f in $RATHENA/conf/*; do
        [ "$(basename $f)" = import ] || ln -s $f $dir/conf/
    done
    if [ "${WORLD_LEAN:-0}" = 1 ]; then
        # Lean world: only the maps in WORLD_MAPS (default: the start point's), their monsters
        # (rAthena's spawn lines for them plus server/npc/*.txt) and the training script, instead
        # of 1265 maps and every NPC.
        local maps=${WORLD_MAPS:-${WORLD_START_POINT%%,*}} m rules
        rules=$(world_rules)
        rm $dir/conf/maps_athena.conf
        for m in $maps; do echo "map: $m"; done > $dir/conf/maps_athena.conf
        mkdir -p $dir/npc/$rules
        for m in $maps; do grep -h "^$m," $RATHENA/npc/$rules/mobs/fields/*.txt; done > $dir/npc/field_mobs.txt
        {
            echo "npc: npc/field_mobs.txt"
            for f in $RG/server/npc/*.txt; do echo "npc: $f"; done
        } > $dir/npc/$rules/scripts_main.conf
    else
        ln -s $RATHENA/npc $dir/npc
    fi
    cp $RATHENA/conf/import-tmpl/* $dir/conf/import/
    {
        for key in login_server ipban_db char_server map_server web_server log_db; do
            echo "${key}_ip: 127.0.0.1"
            echo "${key}_port: $WORLD_DB_PORT"
            echo "${key}_id: ragnarok"
            echo "${key}_pw: $WORLD_DB_PW"
            echo "${key}_db: $dbname"
        done
    } > $dir/conf/import/inter_conf.txt
    cat > $dir/conf/import/login_conf.txt <<EOF
bind_ip: 127.0.0.1
login_port: $lp
new_account: no
ipban_dynamic_pass_failure_ban: no
EOF
    cat > $dir/conf/import/char_conf.txt <<EOF
userid: s1
passwd: $inter_pw
server_name: train$w
bind_ip: 127.0.0.1
login_ip: 127.0.0.1
login_port: $lp
char_ip: 127.0.0.1
char_port: $cp
pincode_enabled: no
start_point: $WORLD_START_POINT
start_point_pre: $WORLD_START_POINT
start_items: $WORLD_START_ITEMS
start_items_pre: $WORLD_START_ITEMS
EOF
    cat > $dir/conf/import/map_conf.txt <<EOF
userid: s1
passwd: $inter_pw
bind_ip: 127.0.0.1
char_ip: 127.0.0.1
char_port: $cp
map_ip: 127.0.0.1
map_port: $mp
${WORLD_MAP_CONF:-}
EOF
    # No SQL logs: the map server writes them synchronously, once per pickup or command.
    printf 'enable_logs: 0\nlog_commands: no\nlog_npc: no\nlog_cash: no\nlog_chat: 0\n' \
        > $dir/conf/import/log_conf.txt
    # Every bot connects from 127.0.0.1; flood protection would ban it after 5 connections in 3 s.
    cat > $dir/conf/import/packet_conf.txt <<EOF
allow: 127.0.0.1
stall_time: 300
EOF
    chmod 600 $dir/conf/import/*
    {
        echo "CREATE DATABASE $dbname;"
        echo "USE $dbname;"
        echo "SOURCE $RATHENA/sql-files/main.sql;"
        echo "SOURCE $RATHENA/sql-files/logs.sql;"
        echo "UPDATE login SET user_pass = '$inter_pw' WHERE account_id = 1 AND userid = 's1';"
        echo "GRANT ALL ON $dbname.* TO 'ragnarok'@'127.0.0.1';"
        local total=$((nbots * WORLD_ACCOUNT_BLOCKS))
        for i in $(seq -f %05g 1 $total); do
            echo "('w${w}b$i', '$bot_pw', 'M', 'a@a.com', 0)"
        done | paste -sd, - | sed "s/^/INSERT INTO login (userid, user_pass, sex, email, group_id) VALUES /; s/$/;/"
        # WORLD_GM_BOT=1 makes the first bot a GM (group 99), for tests that use GM commands.
        if [ "${WORLD_GM_BOT:-0}" = 1 ]; then
            echo "UPDATE login SET group_id = 99 WHERE userid = 'w${w}b00001';"
        fi
    } | wsql || { wlog "world $w: database setup failed"; return 1; }
    for i in $(seq -f %05g 1 $((nbots * WORLD_ACCOUNT_BLOCKS))); do
        echo "w${w}b$i $bot_pw"
    done > $dir/accounts
    echo 0 > $dir/next_block
    chmod 600 $dir/accounts
    wlog "world $w: database $dbname, $((nbots * WORLD_ACCOUNT_BLOCKS)) bot accounts, ports $lp/$cp/$mp"
}

world_wait_port() {
    local port=$1 what=$2
    for i in $(seq 120); do
        ss -ltn | grep -qE "127.0.0.1:$port\b" && return 0
        sleep 0.5
    done
    wlog "$what did not open port $port"
    return 1
}

world_start() {
    local w=$1 dir=$WORLD_ROOT/w$1 t0=$SECONDS
    # ~/ragnarok/lib holds libmysqlclient.so.21 from the login node, for nodes without it.
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$RO/lib:"*) ;;
        *) export LD_LIBRARY_PATH=$RO/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ;;
    esac
    (cd $dir && exec ./login-server > $dir/log/login.out 2>&1) &
    WORLD_PIDS+=($!)
    world_wait_port $((WORLD_PORT_BASE + 10 * w)) "world $w login server" || return 1
    (cd $dir && exec ./char-server > $dir/log/char.out 2>&1) &
    WORLD_PIDS+=($!)
    world_wait_port $((WORLD_PORT_BASE + 10 * w + 1)) "world $w char server" || return 1
    (cd $dir && exec ./map-server > $dir/log/map.out 2>&1) &
    WORLD_PIDS+=($!)
    # "ready" comes before the map server has logged on to the char server; a login in between
    # hangs at character select, so wait for "now online".
    for i in $(seq 240); do
        grep -q "Map Server is now online" $dir/log/map.out 2>/dev/null && break
        sleep 0.5
    done
    if ! grep -q "Map Server is now online" $dir/log/map.out; then
        wlog "world $w map server not ready: $(tail -3 $dir/log/map.out)"
        return 1
    fi
    wlog "world $w up in $((SECONDS - t0))s"
}

world_alive() {
    local pid
    for pid in "${WORLD_PIDS[@]}"; do
        kill -0 $pid 2>/dev/null || return 1
    done
}

# True while process $1 runs (a child that exited stays a zombie until waited for).
world_running() {
    [ -r /proc/$1/status ] && ! grep -q '^State:[[:space:]]*Z' /proc/$1/status 2>/dev/null
}

# Stops process $1 with TERM, and with KILL if it is still running after WORLD_STOP_WAIT seconds:
# a lockstep map server has been seen to spin forever while kicking its last players at shutdown.
world_stop() {
    local pid=$1 deadline=$((SECONDS + ${WORLD_STOP_WAIT:-30}))
    kill -TERM $pid 2>/dev/null
    while world_running $pid && [ $SECONDS -lt $deadline ]; do
        sleep 0.2
    done
    if world_running $pid; then
        wlog "process $pid ($(ps -o comm= -p $pid)) did not stop in ${WORLD_STOP_WAIT:-30}s; killing it"
        kill -KILL $pid 2>/dev/null
    fi
    wait $pid 2>/dev/null
}

world_stop_all() {
    local pid d
    for ((i = ${#WORLD_PIDS[@]} - 1; i >= 0; i--)); do
        world_stop ${WORLD_PIDS[$i]}
    done
    WORLD_PIDS=()
    for d in $WORLD_ROOT/w*; do
        [ -d $d/log ] || continue
        for f in login char map; do
            [ -f $d/log/$f.out ] && cp $d/log/$f.out $WORLD_LOGS/$(basename $d)-$f-$SLURM_JOB_ID${WORLD_LOG_SUFFIX:-}.log
        done
    done
    if [ -n "${WORLD_DB_PID:-}" ]; then
        wdb mariadb-admin --socket=/run/mysqld/mysqld.sock shutdown > /dev/null 2>&1
        wait $WORLD_DB_PID 2>/dev/null
        WORLD_DB_PID=
    fi
    enroot remove -f $WORLD_DB_NAME > /dev/null 2>&1
    rm -rf $WORLD_ROOT
    wlog "worlds stopped"
}

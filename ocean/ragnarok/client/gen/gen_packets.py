#!/usr/bin/env python3
"""Generate the bot's packet table from rAthena's own source, for one PACKETVER.

Usage: python3 gen_packets.py [rathena_dir] [packetver]
Writes ../rathena_packets.h and packets_<ver>.tsv next to this script.

Sources, all from rAthena at the given client version (PRERE):
- headers declared with DEFINE_PACKET_HEADER and their PACKET_* structs (common/packets.hpp,
  map/packets.hpp, map/packets_struct.hpp): length = sizeof the struct, variable when its second
  member is a length field;
- map/clif_packetdb.hpp and map/clif_shuffle.hpp, as the map server loads them in packetdb_readdb.
Where both give a length they must agree, or the script stops.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RA = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "ragnarok/rathena-train")
VER = sys.argv[2] if len(sys.argv) > 2 else "20130618"
FLAGS = [
    "-std=c++17", f"-I{RA}/src/common", f"-I{RA}/src", f"-I{RA}/src/map", f"-I{RA}/3rdparty/libconfig",
    f"-I{RA}/3rdparty/rapidyaml/src", f"-I{RA}/3rdparty/rapidyaml/ext/c4core/src",
    f"-I{RA}/3rdparty/json/include", "-I/usr/include/mysql", "-DHAS_TLS", "-fno-strict-aliasing",
    "-DHAVE_GETADDRINFO", "-DHAVE_SETRLIMIT", "-DHAVE_STRNLEN", f"-DPACKETVER={VER}", "-DSOCKET_EPOLL",
    "-DPRERE", "-DHAVE_MONOTONIC_CLOCK",
]
# common/packets.hpp (login and char server packets) and map/packets.hpp share the include guard
# PACKETS_HPP, so they go through the compiler separately.
UNITS = {
    "char": '#include <common/packets.hpp>\n',
    "map": '#include "map/packets.hpp"\n#include "map/clif.hpp"\n',
}
# Structs mirrored into C for the client, per compile unit, dependencies first.
MIRROR = {
    "char": ["PACKET_CA_LOGIN", "PACKET_AC_ACCEPT_LOGIN_sub", "PACKET_AC_ACCEPT_LOGIN", "PACKET_AC_REFUSE_LOGIN",
             "PACKET_HC_ACCEPT_ENTER2", "CHARACTER_INFO", "PACKET_HC_ACCEPT_ENTER", "PACKET_HC_REFUSE_ENTER",
             "PACKET_HC_CHARLIST_NOTIFY", "PACKET_CH_MAKE_CHAR", "PACKET_HC_ACCEPT_MAKECHAR",
             "PACKET_HC_REFUSE_MAKECHAR", "PACKET_CH_SELECT_CHAR", "PACKET_HC_NOTIFY_ZONESVR", "PACKET_PING"],
    "map": ["EQUIPSLOTINFO", "PACKET_ZC_ACCEPT_ENTER", "PACKET_ZC_NPCACK_MAPMOVE", "PACKET_ZC_NOTIFY_VANISH",
            "PACKET_ZC_NOTIFY_PLAYERMOVE", "PACKET_ZC_STOPMOVE", "packet_spawn_unit", "packet_unit_walking",
            "packet_idle_unit", "PACKET_ZC_NOTIFY_ACT", "PACKET_ZC_ITEM_ENTRY", "packet_dropflooritem",
            "PACKET_ZC_ITEM_DISAPPEAR", "PACKET_ZC_PAR_CHANGE", "PACKET_ZC_LONGPAR_CHANGE",
            "PACKET_ZC_ITEM_PICKUP_ACK", "NORMALITEM_INFO", "EQUIPITEM_INFO", "PACKET_ZC_USE_ITEM_ACK",
            "PACKET_ZC_ITEM_THROW_ACK", "PACKET_ZC_DELETE_ITEM_FROM_BODY", "PACKET_ZC_NOTIFY_TIME",
            "PACKET_ZC_ATTACK_RANGE", "PACKET_ZC_REQ_WEAR_EQUIP_ACK", "PACKET_ZC_STATUS",
            "PACKET_CZ_REQ_WEAR_EQUIP", "PACKET_ZC_NOTIFY_CHAT", "PACKET_ZC_NOTIFY_PLAYERCHAT",
            "PACKET_ZC_SPRITE_CHANGE", "SKILLDATA", "PACKET_ZC_SKILLINFO_LIST", "PACKET_ZC_SKILLINFO_UPDATE",
            "PACKET_ZC_USESKILL_ACK", "PACKET_ZC_ACK_TOUSESKILL", "PACKET_ZC_SKILL_POSTDELAY",
            "PACKET_ZC_NOTIFY_SKILL", "PACKET_ZC_USE_SKILL"],
}
C_TYPES = {"int8": "int8_t", "uint8": "uint8_t", "int16": "int16_t", "uint16": "uint16_t", "int32": "int32_t",
           "uint32": "uint32_t", "int64": "int64_t", "uint64": "uint64_t", "bool": "uint8_t"}
LENGTH_FIELD = re.compile(r"(?i)^(packet)?_?(length|len|size)$")


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout


def struct_body(text, name):
    """The text between the braces of `struct name { ... }`, or None."""
    m = re.search(r"\bstruct\s+%s\s*\{" % re.escape(name), text)
    if not m:
        return None
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i - 1]


def to_c(name, body, mirrored):
    """A packed C typedef RO_<name> for a preprocessed rAthena struct body."""
    body = re.sub(r"\s+", " ", body).strip()
    body = re.sub(r"\bstruct\s*\{", "struct __attribute__((packed)) {", body)
    for cxx, c in C_TYPES.items():
        body = re.sub(r"\b%s\b" % cxx, c, body)
    for other in mirrored:
        body = re.sub(r"(\bstruct\s+)?\b%s\b" % re.escape(other), "RO_" + other, body)
    # C++ has no flexible array members: drop them; callers find the entries after sizeof(head).
    parts = [f.strip() for f in body.split(";") if f.strip()]
    flex = [f for f in parts if f.endswith("[]")]
    parts = [f for f in parts if not f.endswith("[]")]
    fields = "\n".join("    " + f + ";" for f in parts)
    if flex:
        fields += "\n    // followed by: " + "; ".join(flex)
    fields = fields.replace("{;", "{").replace("    } ", "    } ")
    return f"typedef struct __attribute__((packed)) RO_{name} {{\n{fields}\n}} RO_{name};\n"


def struct_members(text, name):
    """Member names of `struct name { ... }` in preprocessed text, top level only, or None."""
    m = re.search(r"\bstruct\s+%s\s*\{" % re.escape(name), text)
    if not m:
        return None
    depth, i, start = 1, m.end(), m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    body, flat, depth = text[start:i - 1], "", 0
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0:
            flat += ch
    names = []
    for decl in flat.split(";"):
        decl = re.sub(r"\[[^\]]*\]", "", decl).strip()
        if decl:
            names.append(re.findall(r"[A-Za-z_]\w*", decl)[-1])
    return names


def dump_unit(gen, unit, includes):
    """Compile and run one dump program; return its output lines."""
    (gen / f"pp_{unit}.cpp").write_text(includes)
    pre = run(["g++", "-E", "-P", *FLAGS, str(gen / f"pp_{unit}.cpp")])
    headers = {}
    for name, value in re.findall(r"const\s+int16\s+HEADER_(\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)\s*;", pre):
        headers[name] = int(value, 0)
    structs = {}
    for name in headers:
        members = struct_members(pre, "PACKET_" + name)
        if members is not None:
            structs[name] = len(members) >= 2 and bool(LENGTH_FIELD.match(members[1]))
    lines = [includes, "#include <cstdio>", "#include <cstddef>",
             "static void add(int cmd, int len, const char* func, const char* pos) { std::printf(\"D %d %d %s %s\\n\", cmd, (short)len, func, pos); }",
             "int main() {"]
    for name in sorted(structs):
        lines.append(f'  std::printf("S %d %zu {int(structs[name])} {name}\\n", (int)HEADER_{name}, sizeof(PACKET_{name}));')
    bodies = {}
    for name in MIRROR[unit]:
        body = struct_body(pre, name)
        if body is None:
            sys.exit(f"{unit}: struct {name} not found")
        bodies[name] = body
        lines.append(f'  std::printf("M 0 %zu {name}\\n", sizeof({name}));')
    if unit == "map":
        # The two packet-db headers define their own macros; drop those lines and use ours.
        for src in ("clif_packetdb.hpp", "clif_shuffle.hpp"):
            text = (RA / "src/map" / src).read_text()
            text = re.sub(r"^\s*#define\s+(packet|parseable_packet)\b.*$", "", text, flags=re.M)
            (gen / src).write_text(text)
        lines += ["#define packet(cmd,length) add(cmd,length,\"-\",\"-\")",
                  "#define parseable_packet(cmd,length,func,...) add(cmd,length,#func,#__VA_ARGS__)",
                  f'#include "{gen}/clif_packetdb.hpp"', f'#include "{gen}/clif_shuffle.hpp"']
    lines.append("}")
    (gen / f"dump_{unit}.cpp").write_text("\n".join(lines) + "\n")
    run(["g++", *FLAGS, "-w", "-o", str(gen / f"dump_{unit}"), str(gen / f"dump_{unit}.cpp")])
    return run([str(gen / f"dump_{unit}")]).splitlines(), bodies


def main():
    gen = HERE / "build"
    gen.mkdir(exist_ok=True)
    out, bodies = [], {}
    for unit, includes in UNITS.items():
        lines, unit_bodies = dump_unit(gen, unit, includes)
        out += lines
        bodies.update(unit_bodies)

    structs, db, sizes = {}, {}, {}  # cmd -> (len, name) / cmd -> (len, func, positions); later db lines win
    for line in out:
        kind, *rest = line.split(" ", 4)
        cmd, length = int(rest[0]), int(rest[1])
        if kind == "M":
            sizes[rest[2]] = length
        elif kind == "S":
            length, name = (-1 if int(rest[2]) else length), rest[3]
            if cmd in structs and structs[cmd][0] != length:
                sys.exit(f"conflict for 0x{cmd:04x}: {name} {length} vs {structs[cmd]}")
            structs[cmd] = (length, name)
        else:
            # rAthena ends the offset list with 0; offset 0 is the packet type, never a field.
            positions = [x for x in "".join(rest[3].split()).split(",") if x not in ("-", "0")]
            db[cmd] = (length, rest[2], positions)

    # Both sources: the map server frames client packets (CZ_) by its packet db, while every server
    # sends its own packets, and the login and char servers parse theirs, by the struct.
    table = {}
    for cmd in sorted(set(structs) | set(db)):
        if cmd in structs and cmd in db and structs[cmd][0] != db[cmd][0]:
            use_db = structs[cmd][1].startswith("CZ_")
            print(f"note: 0x{cmd:04x} {structs[cmd][1]}: struct {structs[cmd][0]}, packet db {db[cmd][0]};"
                  f" using the {'packet db' if use_db else 'struct'}")
            table[cmd] = db[cmd][0] if use_db else structs[cmd][0]
        else:
            table[cmd] = structs[cmd][0] if cmd in structs else db[cmd][0]

    # Client-to-map packet ids by parse function: the last registration that still owns its id.
    parse_ids = {}
    for cmd, (length, func, positions) in db.items():
        if func != "-":
            parse_ids[func] = (cmd, length, positions)

    tsv = HERE / f"packets_{VER}.tsv"
    with tsv.open("w") as f:
        for cmd in table:
            name = structs[cmd][1] if cmd in structs else (db[cmd][1] if db.get(cmd, (0, "-"))[1] != "-" else "")
            f.write(f"0x{cmd:04x}\t{table[cmd]}\t{name}\n")
    with (HERE.parent / "rathena_packets.h").open("w") as f:
        f.write(f"// Generated by gen/gen_packets.py from rAthena for PACKETVER {VER} (pre-renewal). Do not edit.\n")
        f.write("#pragma once\n#include <stdint.h>\n\n")
        f.write("// Valid as C11 and as C++ (the PufferLib trainer compiles envs with nvcc as C++17).\n")
        f.write("#ifdef __cplusplus\n#define RO_STATIC_ASSERT static_assert\n#else\n#define RO_STATIC_ASSERT _Static_assert\n#endif\n\n")
        f.write(f"#define RO_PACKETVER {VER}\n\n")
        f.write("// Client-to-map packets as the map server parses them: id, length, field offsets.\n")
        for func in sorted(parse_ids):
            cmd, length, positions = parse_ids[func]
            macro = "RO_CZ_" + func.removeprefix("clif_parse_").upper()
            f.write(f"#define {macro} 0x{cmd:04x} // length {length}; fields at {','.join(positions) or '-'}\n")
            f.write(f"#define {macro}_LEN {length}\n")
            for k, pos in enumerate(positions):
                f.write(f"#define {macro}_P{k} {pos}\n")
        f.write("\n// Packet structs, mirrored from rAthena's definitions for this version.\n")
        mirrored = [n for unit in MIRROR.values() for n in unit]
        for name in mirrored:
            f.write(to_c(name, bodies[name], mirrored))
            f.write(f'RO_STATIC_ASSERT(sizeof(RO_{name}) == {sizes[name]}, "RO_{name} size");\n\n')
        f.write("\n// Packet lengths, both directions: 0 = unknown, -1 = variable (length at offset 2).\n")
        f.write("static inline int ro_packet_len(int cmd) {\n    switch (cmd) {\n")
        for cmd in table:
            f.write(f"    case 0x{cmd:04x}: return {table[cmd]};\n")
        f.write("    default: return 0;\n    }\n}\n")
    print(f"{len(table)} packets ({len(structs)} from structs, {len(parse_ids)} client-to-map ids,"
          f" {len(bodies)} mirrored structs)"
          f" -> {tsv.name}, rathena_packets.h")


if __name__ == "__main__":
    main()

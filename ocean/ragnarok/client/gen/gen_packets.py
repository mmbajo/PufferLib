#!/usr/bin/env python3
"""Generate the bot's packet table from rAthena's own source, for one PACKETVER and rule set.

Usage: python3 gen_packets.py [rathena_dir] [packetver] [pre-re|re] [out.h]
Writes ../rathena_packets.h (or out.h) and packets_<ver>.tsv next to this script. The version and
the rules default to those the checkout was configured with (its src/map/Makefile), else 20130618
pre-renewal.

Sources, all from rAthena at the given client version and rules:
- headers declared with DEFINE_PACKET_HEADER and their PACKET_* structs (common/packets.hpp,
  map/packets.hpp, map/packets_struct.hpp): length = sizeof the struct, variable when its second
  member is a length field;
- map/clif_packetdb.hpp and map/clif_shuffle.hpp, as the map server loads them in packetdb_readdb;
- the version-dependent ids in map/packets_struct.hpp's enum packet_headers (unit packets, damage,
  inventory lists...), written as RO_TYPE_<name>.
Every header is written as RO_HEADER_<NAME>, and every member of a mirrored struct as
RO_HAS_<struct>_<member>, so the client can follow ids and layouts that move between versions.
Where both give a length they must agree, or the script stops.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RA = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "ragnarok/rathena-2025")


def build_settings(ra):
    """(PACKETVER, rules) the checkout was configured with, from its src/map/Makefile, or Nones."""
    mk = ra / "src/map/Makefile"
    if not mk.exists():
        return None, None
    text = mk.read_text()
    m = re.search(r"-DPACKETVER=(\d+)", text)
    return (m.group(1) if m else None), ("pre-re" if "-DPRERE" in text else "re")


BUILT_VER, BUILT_RULES = build_settings(RA)
VER = sys.argv[2] if len(sys.argv) > 2 else BUILT_VER or "20130618"
RULES = sys.argv[3] if len(sys.argv) > 3 else BUILT_RULES or "pre-re"
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else HERE.parent / "rathena_packets.h"
if RULES not in ("pre-re", "re"):
    sys.exit(f"rules must be pre-re or re, not {RULES}")
if (BUILT_VER, BUILT_RULES) not in ((None, None), (VER, RULES)):
    print(f"note: {RA.name} is built for {BUILT_VER} {BUILT_RULES}, generating for {VER} {RULES}")
FLAGS = [
    "-std=c++17", f"-I{RA}/src/common", f"-I{RA}/src", f"-I{RA}/src/map", f"-I{RA}/3rdparty/libconfig",
    f"-I{RA}/3rdparty/rapidyaml/src", f"-I{RA}/3rdparty/rapidyaml/ext/c4core/src",
    f"-I{RA}/3rdparty/json/include", "-I/usr/include/mysql", "-DHAS_TLS", "-fno-strict-aliasing",
    "-DHAVE_GETADDRINFO", "-DHAVE_SETRLIMIT", "-DHAVE_STRNLEN", f"-DPACKETVER={VER}", "-DSOCKET_EPOLL",
    *(["-DPRERE"] if RULES == "pre-re" else []), "-DHAVE_MONOTONIC_CLOCK",
]
# common/packets.hpp (login and char server packets) and map/packets.hpp share the include guard
# PACKETS_HPP, so they go through the compiler separately.
UNITS = {
    "char": '#include <common/packets.hpp>\n',
    "map": '#include "map/packets.hpp"\n#include "map/clif.hpp"\n',
}
# Structs mirrored into C for the client, per compile unit, dependencies first. A leading '?' marks
# a struct that only some versions have; it is mirrored when present (RO_HAVE_<name>).
MIRROR = {
    "char": ["PACKET_CA_LOGIN", "PACKET_AC_ACCEPT_LOGIN_sub", "PACKET_AC_ACCEPT_LOGIN", "PACKET_AC_REFUSE_LOGIN",
             "PACKET_HC_ACCEPT_ENTER2", "CHARACTER_INFO", "PACKET_HC_ACCEPT_ENTER", "PACKET_HC_REFUSE_ENTER",
             "PACKET_HC_CHARLIST_NOTIFY", "PACKET_CH_MAKE_CHAR", "PACKET_HC_ACCEPT_MAKECHAR",
             "PACKET_HC_REFUSE_MAKECHAR", "PACKET_CH_SELECT_CHAR", "PACKET_HC_NOTIFY_ZONESVR", "PACKET_PING"],
    "map": ["EQUIPSLOTINFO", "PACKET_ZC_ACCEPT_ENTER", "PACKET_ZC_NPCACK_MAPMOVE", "PACKET_ZC_NOTIFY_VANISH",
            "PACKET_ZC_NOTIFY_PLAYERMOVE", "PACKET_ZC_STOPMOVE", "packet_spawn_unit", "packet_unit_walking",
            "packet_idle_unit", "PACKET_ZC_NOTIFY_ACT", "PACKET_ZC_ITEM_ENTRY", "packet_dropflooritem",
            "PACKET_ZC_ITEM_DISAPPEAR", "PACKET_ZC_PAR_CHANGE", "PACKET_ZC_LONGPAR_CHANGE",
            "?ItemOptions", "PACKET_ZC_ITEM_PICKUP_ACK", "NORMALITEM_INFO", "EQUIPITEM_INFO", "PACKET_ZC_USE_ITEM_ACK",
            "PACKET_ZC_ITEM_THROW_ACK", "PACKET_ZC_DELETE_ITEM_FROM_BODY", "PACKET_ZC_NOTIFY_TIME",
            "PACKET_ZC_ATTACK_RANGE", "PACKET_ZC_REQ_WEAR_EQUIP_ACK", "PACKET_ZC_STATUS",
            "PACKET_CZ_REQ_WEAR_EQUIP", "PACKET_ZC_NOTIFY_CHAT", "PACKET_ZC_NOTIFY_PLAYERCHAT",
            "PACKET_ZC_SPRITE_CHANGE", "SKILLDATA", "PACKET_ZC_SKILLINFO_LIST", "PACKET_ZC_SKILLINFO_UPDATE",
            "PACKET_ZC_USESKILL_ACK", "PACKET_ZC_ACK_TOUSESKILL", "PACKET_ZC_SKILL_POSTDELAY",
            "PACKET_ZC_NOTIFY_SKILL", "PACKET_ZC_USE_SKILL", "?PACKET_ZC_LONGLONGPAR_CHANGE"],
}
# Server packets whose id comes from enum packet_headers, and the struct clif.cpp sends with it. Their
# length is that struct's size: the packet db's entry for the id can be stale (it is only used to
# parse what clients send), e.g. 0x0add is 22 there but 24 bytes on the wire from 2018-11 on.
TYPED = {
    "dropflooritemType": "packet_dropflooritem", "sendLookType": "PACKET_ZC_SPRITE_CHANGE",
    "useItemAckType": "PACKET_ZC_USE_ITEM_ACK", "graffiti_entryType": "packet_graffiti_entry",
    "skillscale": "PACKET_ZC_SKILL_SCALE", "buyingStoreUpdateItemType": "PACKET_ZC_UPDATE_ITEM_FROM_BUYING_STORE",
    "rouletteinfoackType": "packet_roulette_info_ack", "roulettgenerateackType": "packet_roulette_generate_ack",
    "roulettercvitemackType": "packet_roulette_itemrecv_ack",
}
# Offsets the dump program measures: name -> (struct, member).
OFFSETS = {
    "RO_ITEMLIST_NORMAL_HEAD": ("packet_itemlist_normal", "list"),
    "RO_ITEMLIST_EQUIP_HEAD": ("packet_itemlist_equip", "list"),
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
    enum = struct_body(pre.replace("enum packet_headers", "struct packet_headers"), "packet_headers") or ""
    enum_names = re.findall(r"(\w+)\s*=", enum)
    lines = [includes, "#include <cstdio>", "#include <cstddef>",
             "static void add(int cmd, int len, const char* func, const char* pos) { std::printf(\"D %d %d %s %s\\n\", cmd, (short)len, func, pos); }",
             "int main() {"]
    for name in sorted(structs):
        lines.append(f'  std::printf("S %d %zu {int(structs[name])} {name}\\n", (int)HEADER_{name}, sizeof(PACKET_{name}));')
    bodies = {}
    for entry in MIRROR[unit]:
        name = entry.lstrip("?")
        body = struct_body(pre, name)
        if body is None:
            if entry.startswith("?"):
                continue
            sys.exit(f"{unit}: struct {name} not found")
        bodies[name] = body
        lines.append(f'  std::printf("M 0 %zu {name}\\n", sizeof({name}));')
    for name in headers:
        lines.append(f'  std::printf("H %d 0 {name}\\n", (int)HEADER_{name});')
    for name in enum_names:
        lines.append(f'  std::printf("T %d 0 {name}\\n", (int){name});')
    if unit == "map":
        for type_name, struct in TYPED.items():
            members = struct_members(pre, struct)
            if type_name in enum_names and members is not None:
                var = int(len(members) >= 2 and bool(LENGTH_FIELD.match(members[1])))
                lines.append(f'  std::printf("Y %d %zu {var} {type_name} {struct}\\n", (int){type_name}, sizeof(struct {struct}));')
        for macro, (struct, member) in OFFSETS.items():
            lines.append(f'  std::printf("O %zu 0 {macro}\\n", offsetof(struct {struct}, {member}));')
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
    header_ids, type_ids, offsets, typed = {}, {}, {}, {}
    for line in out:
        kind, *rest = line.split(" ", 4)
        cmd, length = int(rest[0]), int(rest[1])
        if kind == "M":
            sizes[rest[2]] = length
        elif kind == "H":
            if header_ids.get(rest[2], cmd) != cmd:
                sys.exit(f"HEADER_{rest[2]} is 0x{header_ids[rest[2]]:04x} in one unit and 0x{cmd:04x} in the other")
            header_ids[rest[2]] = cmd
        elif kind == "T":
            type_ids[rest[2]] = cmd
        elif kind == "O":
            offsets[rest[2]] = cmd
        elif kind == "Y":
            typed[cmd] = (-1 if int(rest[2]) else length, *rest[3].split())
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
    for cmd, (length, type_name, struct) in typed.items():
        if table.get(cmd, length) != length:
            print(f"note: 0x{cmd:04x} {type_name}: packet db {table[cmd]}, {struct} {length}; using the struct")
        table[cmd] = length

    # Client-to-map packet ids by parse function: the last registration that still owns its id.
    parse_ids = {}
    for cmd, (length, func, positions) in db.items():
        if func != "-":
            parse_ids[func] = (cmd, length, positions)

    tsv = HERE / f"packets_{VER}_{RULES}.tsv"
    with tsv.open("w") as f:
        for cmd in table:
            name = structs[cmd][1] if cmd in structs else (db[cmd][1] if db.get(cmd, (0, "-"))[1] != "-" else "")
            f.write(f"0x{cmd:04x}\t{table[cmd]}\t{name}\n")
    rules_name = "renewal" if RULES == "re" else "pre-renewal"
    with OUT.open("w") as f:
        f.write(f"// Generated by gen/gen_packets.py from rAthena for PACKETVER {VER} ({rules_name}). Do not edit.\n")
        f.write("#pragma once\n#include <stdint.h>\n\n")
        f.write("// Valid as C11 and as C++ (the PufferLib trainer compiles envs with nvcc as C++17).\n")
        f.write("#ifdef __cplusplus\n#define RO_STATIC_ASSERT static_assert\n#else\n#define RO_STATIC_ASSERT _Static_assert\n#endif\n\n")
        f.write(f"#define RO_PACKETVER {VER}\n")
        f.write(f"#define RO_RENEWAL {int(RULES == 're')}  // the server's rules: its db/ and npc/ subfolder is {RULES}\n")
        f.write(f"#define RO_RULES_DIR \"{RULES}\"\n\n")
        f.write("// Packet ids by rAthena's name for this version (DEFINE_PACKET_HEADER).\n")
        for name in sorted(header_ids):
            f.write(f"#define RO_HEADER_{name} 0x{header_ids[name] & 0xffff:04x}\n")
        f.write("\n// Ids that rAthena picks by version in enum packet_headers (map/packets_struct.hpp).\n")
        for name in sorted(type_ids):
            f.write(f"#define RO_TYPE_{name} 0x{type_ids[name] & 0xffff:04x}\n")
        f.write("\n// Offsets of the first entry in list packets.\n")
        for name in sorted(offsets):
            f.write(f"#define {name} {offsets[name]}\n")
        f.write("\n")
        f.write("// Client-to-map packets as the map server parses them: id, length, field offsets.\n")
        for func in sorted(parse_ids):
            cmd, length, positions = parse_ids[func]
            macro = "RO_CZ_" + func.removeprefix("clif_parse_").upper()
            f.write(f"#define {macro} 0x{cmd:04x} // length {length}; fields at {','.join(positions) or '-'}\n")
            f.write(f"#define {macro}_LEN {length}\n")
            for k, pos in enumerate(positions):
                f.write(f"#define {macro}_P{k} {pos}\n")
        f.write("\n// Packet structs, mirrored from rAthena's definitions for this version.\n")
        mirrored = [n.lstrip("?") for unit in MIRROR.values() for n in unit if n.lstrip("?") in bodies]
        for name in mirrored:
            f.write(to_c(name, bodies[name], mirrored))
            f.write(f'RO_STATIC_ASSERT(sizeof(RO_{name}) == {sizes[name]}, "RO_{name} size");\n')
            f.write(f"#define RO_HAVE_{name} 1\n")
            for member in struct_members("struct %s {%s}" % (name, bodies[name]), name) or []:
                f.write(f"#define RO_HAS_{name}_{member} 1\n")
            f.write("\n")
        f.write("\n// Packet lengths, both directions: 0 = unknown, -1 = variable (length at offset 2).\n")
        f.write("static inline int ro_packet_len(int cmd) {\n    switch (cmd) {\n")
        for cmd in table:
            f.write(f"    case 0x{cmd:04x}: return {table[cmd]};\n")
        f.write("    default: return 0;\n    }\n}\n")
    print(f"PACKETVER {VER} {RULES}: {len(table)} packets ({len(structs)} from structs, {len(parse_ids)}"
          f" client-to-map ids, {len(header_ids)} headers, {len(type_ids)} enum ids, {len(bodies)} mirrored"
          f" structs) -> {tsv.name}, {OUT.name}")


if __name__ == "__main__":
    main()

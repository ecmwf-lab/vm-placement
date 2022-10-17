#!/usr/bin/env python3
# (C) Copyright 2022- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import argparse
import logging
import os
import random
import re
import socket
from collections import defaultdict

import numpy as np
import openstack
import yaml
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOG = logging.getLogger(__name__)


API = openstack.connect()

PROJECTS = {}
USERS = {}

with open(os.path.join(os.path.dirname(__file__), "vm-placement.yaml")) as f:
    CONFIG = yaml.safe_load(f)


parser = argparse.ArgumentParser()

group = parser.add_argument_group("data collection options")
group.add_argument("--uptime", action="store_true")
group.add_argument("--diagnostics", action="store_true")
group.add_argument("--detailed", action="store_true")

group = parser.add_argument_group("display options")

group.add_argument(
    "--sort",
    choices=[
        "default",
        "read",
        "write",
        "rx",
        "tx",
        "memory_free",
        "vcpus_used",
        "load",
    ],
    default="default",
)
group.add_argument("--reverse", action="store_true")

parser.add_argument("--stopped", action="store_true", help="List stopped VMs")
parser.add_argument("--status")


parser.add_argument("--csv", action="store_true")

parser.add_argument("--list", action="store_true")
parser.add_argument("--placement", action="store_true")
parser.add_argument("--make-space", type=int, default=0)
parser.add_argument("--doit", action="store_true")


parser.add_argument("--nowait", action="store_true")
group = parser.add_argument_group("fitness options")
group.add_argument("--moves-weight", type=int, default=1)
parser.add_argument("--head-room", type=int, default=0)
group = parser.add_argument_group("pygad options")
group.add_argument("--generations", type=int, default=100)
group.add_argument("--population", type=int, default=1000)
group.add_argument("--random", action="store_true")

group.add_argument("--plot", action="store_true")
group.add_argument(
    "--crossover-type",
    choices=["single_point", "two_points", "uniform", "scattered"],
    default="single_point",
)
group.add_argument(
    "--mutation-type",
    choices=["random", "swap", "inversion", "scramble", "adaptive"],
    default="random",
)
group.add_argument(
    "--parent-selection-type",
    choices=["sss", "rws", "sus", "rank", "random", "tournament"],
    default="sss",
)


ARGS = parser.parse_args()


def bytes_to_string(n, zero="0"):

    if n == 0:
        return zero

    if n < 0:
        return f"-{bytes_to_string(-n)}"

    u = ["", "K", "M", "G", "T", "P"]
    i = 0
    while n >= 1024:
        n /= 1024.0
        i += 1
    return "%g%s" % (int(n * 10 + 0.5) / 10.0, u[i])


def round_up(x, size=1024 * 1024 * 1024):
    return int((x + size - 1) / size) * size


def round_down(x, size=1024 * 1024 * 1024):
    return int(x / size) * size


class Wrapper:
    def __init__(self, openstack_object):
        self._openstack_object = openstack_object
        for k, v in openstack_object.toDict().items():
            setattr(self, k, v)


class Hypervisor(Wrapper):
    def __init__(self, hypervisor):
        super().__init__(hypervisor)
        self.servers = []

        self.memory_size *= 1024 * 1024
        self.memory_used *= 1024 * 1024
        self.memory_free *= 1024 * 1024

        if ARGS.uptime:
            self.uptime = (
                API.compute.get_hypervisor_uptime(self.id)["uptime"]
                .strip()
                .split(" ", 1)[1]
            )

            m = re.search(r"load average: ([\d\.]+), ([\d\.]+), ([\d\.]+)", self.uptime)
            if m:
                self.load_average = tuple(float(m.group(x)) for x in (1, 2, 3))
            else:
                self.load_average = ("?", "?", "?")

            m = re.search(r"up (\d+ days?)", self.uptime)

            if m.group(1):
                self.up = m.group(1)
            else:
                self.up = "?"

            self.load = max(self.load_average)
        else:
            self.load_average = ("?", "?", "?")
            self.up = "?"
            self.load = 0

    def append(self, server):
        self.servers.append(server)

    @property
    def default(self):
        return self.name

    @property
    def rx(self):
        return sum(s.rx for s in self.servers)

    @property
    def tx(self):
        return sum(s.tx for s in self.servers)

    @property
    def read(self):
        return sum(s.read for s in self.servers)

    @property
    def write(self):
        return sum(s.write for s in self.servers)

    @property
    def io(self):
        return self.tx + self.rx + self.read + self.write


class Server(Wrapper):
    def __init__(self, server):
        super().__init__(server)
        self._diagnostics = None

        # Compatibility
        if hasattr(self, 'hypervisor_hostname'):
            self.host = self.hypervisor_hostname

        if "id" in self.flavor and self.flavor["id"] in FLAVORS:
            self.flavor = FLAVORS[self.flavor["id"]]
            self.memory_size = self.flavor.ram * 1024 * 1024
            self.vcpus = self.flavor.vcpus
        else:
            self.memory_size = self.flavor.get("ram", 0) * 1024 * 1024
            self.vcpus = self.flavor.get("vcpus", 0)

        self.rx = 0
        self.tx = 0
        self.read = 0
        self.write = 0

        if ARGS.diagnostics:
            for k, v in self.diagnostics.items():
                if k.endswith("_rx"):
                    self.rx += v

                if k.endswith("_tx"):
                    self.tx += v

                if k.endswith("_read"):
                    self.read += v

                if k.endswith("_write"):
                    self.write += v

    @property
    def project_name(self):
        p = PROJECTS.get(self.project_id, self.project_id)
        if isinstance(p, str):
            return p
        return p.name

    @property
    def user_name(self):
        p = USERS.get(self.user_id, self.user_id)
        if isinstance(p, str):
            return p
        return p.name

    @property
    def flavor_name(self):
        assert "id" not in self.flavor
        return self.flavor["original_name"]

    @property
    def image_name(self):
        p = IMAGES.get(self.image["id"], self.image["id"])
        if p is None:
            # assert False, (self.image, IMAGES)
            return "?"
        if isinstance(p, str):
            return p
        return p.name

    @property
    def floating(self):
        return self.accessIPv4

    @property
    def dns(self):
        f = self.floating
        if f != "":
            dns = socket.getnameinfo((f, 0), 0)[0]
            if dns != f:
                return dns

        return ""

    @property
    def comment(self):
        d = self.dns
        if d == "":
            return ""
        try:
            h = socket.gethostbyname(self.dns)
        except socket.error:
            return f"Error resolving {self.dns}"
        if h != self.floating:
            return f"DNS mismatch {h}"
        return ""

    @property
    def diagnostics(self):
        if self._diagnostics is None:
            LOG.info(f"Getting diagnostics for {self.id}")
            self._diagnostics = API.compute.get(
                f"/servers/{self.id}/diagnostics"
            ).json()
        return self._diagnostics


class Project(Wrapper):
    pass


class User(Wrapper):
    pass


class Image(Wrapper):
    pass


LOG.info("Get list of projects")
PROJECTS = {h.id: Project(h) for h in API.list_projects()}
LOG.info(f"Found {len(PROJECTS)} project(s)")

LOG.info("Get list of users")
USERS = {h.id: Project(h) for h in API.list_users()}
LOG.info(f"Found {len(USERS)} user(s)")

LOG.info("Get list of image")
IMAGES = {h.id: Image(h) for h in API.list_images()}
LOG.info(f"Found {len(IMAGES)} image(s)")

LOG.info("Get list of flavors")
FLAVORS = {h.id: Image(h) for h in API.list_flavors()}
LOG.info(f"Found {len(FLAVORS)} flavor(s)")

LOG.info("Get list of hypervisors")
HYPERVISORS = {h.name: Hypervisor(h) for h in API.list_hypervisors()}
LOG.info(f"Found {len(HYPERVISORS)} hypervisor(s)")

LOG.info("Get list of servers (this may take a while)")
SERVERS = [
    Server(h) for h in API.list_servers(all_projects=True, detailed=ARGS.detailed)
]
LOG.info(f"Found {len(SERVERS)} server(s)")

if not ARGS.stopped and not ARGS.csv:
    HYPERVISORS.pop(None, None)

HOSTS = list(HYPERVISORS.values())

hosts_len = max(len(x.name) for x in HOSTS)
names_len = 0
for s in SERVERS:
    if s.host in HYPERVISORS:
        HYPERVISORS[s.host].append(s)
    else:
        LOG.warning(f"No hypervisors found [{s.host}] for server {s.name}")
    names_len = max(names_len, len(s.name))


hosts_len = max(hosts_len, names_len) + 3


def sorter(x):
    return getattr(x, ARGS.sort)


def display(HOSTS):
    total = 0
    free = 0
    total_cpus = 0
    total_cpus_used = 0
    for host in sorted(
        HOSTS,
        key=lambda x: sorter(x),
        reverse=ARGS.reverse,
    ):
        total += host.memory_size
        free += host.memory_free
        print(
            host.name.ljust(hosts_len),
            bytes_to_string(host.memory_size),
            "free",
            bytes_to_string(host.memory_free),
            "vcpus",
            host.vcpus,
            "used",
            host.vcpus_used,
            "up",
            host.up,
            "load average %s, %s, %s" % host.load_average,
        )

        total_cpus += host.vcpus
        total_cpus_used += host.vcpus_used
        print(
            " ",
            "".ljust(names_len),
            "mem".rjust(4),
            "cpu".rjust(3),
            "rd".rjust(7),
            "wr".rjust(7),
            "rx".rjust(7),
            "tx".rjust(7),
        )

        for vm in sorted(host.servers, key=lambda x: x.name):
            print(
                " ",
                vm.name.ljust(names_len),
                bytes_to_string(vm.memory_size).rjust(4),
                str(vm.vcpus).rjust(3),
                bytes_to_string(vm.read, "-").rjust(7),
                bytes_to_string(vm.write, "-").rjust(7),
                bytes_to_string(vm.rx, "-").rjust(7),
                bytes_to_string(vm.tx, "-").rjust(7),
                vm.id,
                vm.status,
            )

        print(
            " ",
            "".ljust(names_len),
            "".rjust(4),
            "".rjust(3),
            bytes_to_string(host.read, "-").rjust(7),
            bytes_to_string(host.write, "-").rjust(7),
            bytes_to_string(host.rx, "-").rjust(7),
            bytes_to_string(host.tx, "-").rjust(7),
            bytes_to_string(host.io, "-"),
        )
        print()

    print("----")
    print(
        "Total",
        bytes_to_string(total),
        ", free",
        bytes_to_string(free),
        ", CPUs",
        total_cpus,
        ", CPUs used",
        total_cpus_used,
    )


HYPERVISORS = list(HYPERVISORS.values())
VMS = []
for host in HYPERVISORS:
    VMS += host.servers

if ARGS.csv:
    print(
        "project",
        "name",
        "status",
        "user",
        "memory_size",
        "vcpus",
        "accessIPv4",
        "dns",
        "created",
        "updated",
        "description",
        "id",
        "flavor",
        "image",
        "comment",
        sep=",",
    )

    for vm in sorted(VMS, key=lambda x: (x.status, x.project_name, x.name)):
        if ARGS.status is not None:
            if vm.status.lower() != ARGS.status.lower():
                continue
        # print(dir(vm))
        print(
            vm.project_name,
            vm.name,
            vm.status,
            vm.user_name,
            vm.memory_size,
            vm.vcpus,
            vm.floating,
            vm.dns,
            vm.created,
            vm.updated,
            vm.description,
            vm.id,
            vm.flavor_name,
            vm.image_name,
            vm.comment,
            sep=",",
        )
    exit(0)


if ARGS.list:

    for vm in sorted(VMS, key=lambda x: (x.group, x.name)):
        print(vm.group, vm.name, f"({vm.description})", sep="\t")
    exit(0)


AFFINITY = {
    "always_with": 1,
    "never_with": -1,
}


def affinity(a, b):
    name_a = a.name
    name_b = b.name

    for k, v in CONFIG["affinity"].items():
        ma = re.match(f"^{k}$", name_a)
        if ma:
            for position in ("always_with", "never_with"):
                if position in v:
                    mb = re.match(f"^{v[position]}$", name_b)
                    if mb:
                        if ma.groups() == mb.groups():
                            return AFFINITY[position]

    return 0


class GAHost:
    def __init__(self, host):
        self._host = host

        self.default = self._host.default

        self.vcpus = self._host.vcpus
        self.name = self._host.name

        self.memory_size = round_down(self._host.memory_size)

        self.number = int(re.search(r"(\d+)", self.name).group(1))

    def finalise(self):

        self.memory_free = self.memory_size
        self.memory_used = 0
        self.vcpus_used = 0
        self.rx = 0
        self.tx = 0
        self.read = 0
        self.write = 0

        for s in self.servers:
            memory_size = round_up(s.memory_size)
            self.memory_used += memory_size
            self.memory_free -= memory_size
            self.vcpus_used += s.vcpus
            self.rx += s.rx
            self.tx += s.tx
            self.read += s.read
            self.write += s.write

        self.io = self.tx + self.rx + self.read + self.write


class Cost:
    pass


def placement():
    import pygad

    num_hosts = len(HYPERVISORS)
    num_vms = len(VMS)
    head_room = ARGS.head_room * 1024 * 1024 * 1024
    if head_room:
        LOG.info(f"Headroom {bytes_to_string(head_room)}")

    make_space = ARGS.make_space * 1024 * 1024 * 1024
    if make_space:
        LOG.info(f"Make space for {bytes_to_string(make_space)}")

    num_genes = num_vms
    moves_weight = ARGS.moves_weight

    current = np.zeros(num_genes, dtype=np.int64)
    v = 0
    for i, target in enumerate(HYPERVISORS):
        for vm in target.servers:
            current[v] = i
            v += 1

    love = set()
    hate = set()
    ga_hosts = [GAHost(h) for h in HYPERVISORS]

    for i, v in enumerate(VMS):
        for j, w in enumerate(VMS):
            if j > i:
                a = affinity(v, w)
                if a > 0:
                    love.add((i, j))
                if a < 0:
                    hate.add((i, j))
                a = affinity(w, v)
                if a > 0:
                    love.add((i, j))
                if a < 0:
                    hate.add((i, j))
    print("num_hosts", num_hosts)
    print("num_vms", num_vms)

    def placement_cost(solution, report=False):
        hard_violations = 0
        soft_violations = 0
        location = defaultdict(set)
        collocated = set()
        moves = 0
        appart = 0
        same = 0
        extra_cpus = []
        cpus = []

        for a, b in zip(current, solution):
            if a != b:
                moves += 1

        for h in ga_hosts:
            h.servers = []

        for vm, host in enumerate(solution):
            ga_hosts[host].servers.append(VMS[vm])

            others = location[host]
            for o in others:
                if o < vm:
                    collocated.add((o, vm))
                else:
                    collocated.add((vm, o))
            location[host].add(vm)

        for h in ga_hosts:
            h.finalise()

        for h in ga_hosts:

            if h.memory_free < head_room:
                hard_violations += 1

            cpus.append(h.vcpus_used)

            if h.vcpus_used > h.vcpus:
                extra_cpus.append(h.vcpus_used - h.vcpus)
            else:
                extra_cpus.append(0)

        soft_violations += np.std(cpus)

        for h in hate:
            if h in collocated:
                hard_violations += 1
                same += 1

        for h in love:
            if h not in collocated:
                hard_violations += 1
                appart += 1

        ##########################################################
        movements = []
        for i, (a, b) in enumerate(zip(current, solution)):
            if a != b and VMS[i].memory_size != 0:
                movements.append(
                    (
                        VMS[i].memory_size,
                        ga_hosts[b].name,
                        VMS[i].host,
                    )
                )
        sizes = {h.name: h.memory_free for h in ga_hosts}

        for memory, target, source in movements:
            sizes[source] += memory
            sizes[target] -= memory

        for s in sizes.values():
            if s < 0:
                hard_violations += 1

        ##########################################################

        if report:
            print("moves", moves)
            print("cpus", max(extra_cpus), np.std(cpus), extra_cpus)
            print("appart", appart, len(love))
            for h in love:
                if h not in collocated:
                    print("  Not collocated", VMS[h[0]].name, VMS[h[1]].name)
            print("same", same, len(hate))
            for h in hate:
                if h in collocated:
                    print("  Not separated", VMS[h[0]].name, VMS[h[1]].name)

        return hard_violations * 100000 + soft_violations * 1000 + moves * moves_weight

    def make_space_cost(solution, report=False):
        moves = 0
        soft_violations = 0

        for a, b in zip(current, solution):
            if a != b:
                moves += 1

        hard_violations = 0
        for h in ga_hosts:
            h.servers = []

        for vm, host in enumerate(solution):
            ga_hosts[host].servers.append(VMS[vm])

        for h in ga_hosts:
            h.finalise()

        for h in ga_hosts:
            if h.memory_free < head_room:
                hard_violations += 1

            if h.vcpus_used > 2 * h.vcpus:
                hard_violations += 1

        ##########################################################
        hosts_with_space = 0
        for h in ga_hosts:

            if h.memory_free >= make_space + head_room:
                hosts_with_space += 1

        if hosts_with_space == 0:
            hard_violations += 1000

        ##########################################################
        movements = []
        for i, (a, b) in enumerate(zip(current, solution)):
            if a != b and VMS[i].memory_size != 0:
                movements.append(
                    (
                        VMS[i].memory_size,
                        ga_hosts[b].name,
                        VMS[i].host,
                    )
                )
        sizes = {h.name: h.memory_free for h in ga_hosts}

        for memory, target, source in movements:
            sizes[source] += memory
            sizes[target] -= memory

        for s in sizes.values():
            if s < 0:
                hard_violations += 1

        ##########################################################

        if report:
            print("moves", moves)
            print("hosts with space", hosts_with_space)

        return hard_violations * 100000 + soft_violations * 10 + moves * moves_weight

    cost = make_space_cost if ARGS.make_space else placement_cost
    print("COST", cost)

    def fitness(solution, solution_idx):
        return -cost(solution)

    print("Before")
    c = cost(current, True)
    print("cost", c)
    # Start with current position so we minimize the moves
    pop_size = ARGS.population

    population = np.zeros((pop_size, num_genes))
    for i in range(pop_size):
        if i == 0:
            population[i, :] = current
        else:
            population[i, :] = current
            random.shuffle(population[i, :])

    if ARGS.random:
        population = None

    num_generations = ARGS.generations
    with tqdm(total=num_generations) as pbar:
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=max(2, int(pop_size * 0.05)),
            fitness_func=fitness,
            num_genes=num_genes,
            sol_per_pop=pop_size,
            initial_population=population,
            parent_selection_type=ARGS.parent_selection_type,
            crossover_type=ARGS.crossover_type,
            mutation_type=ARGS.mutation_type,
            mutation_num_genes=3,
            mutation_by_replacement=True,
            gene_type=int,
            gene_space=dict(low=0, high=num_hosts),
            on_generation=lambda _: pbar.update(1),
        )
        ga_instance.run()

    solution, _, _ = ga_instance.best_solution()

    print("After")

    c = cost(solution, True)
    print("cost", c)
    # print(solution)

    if ARGS.plot:
        ga_instance.plot_fitness()

    #
    moves = []
    sizes = {h.name: h.memory_free for h in ga_hosts}
    # LOG.info("SIZE %s", sizes)

    for i, (a, b) in enumerate(zip(current, solution)):
        if a != b:
            if VMS[i].status != "ACTIVE":
                LOG.warning(f"Ignoring '{VMS[i].name}' with status {VMS[i].status}")
                continue
            moves.append((VMS[i], ga_hosts[b].name))

    for server, target in moves:
        id, name, source, memory = (
            server.id,
            server.name,
            server.host,
            server.memory_size,
        )
        LOG.info(
            (
                f"Moving {name} (mem={bytes_to_string(memory)}) "
                f'{source.split(".")[0]} (free={bytes_to_string(sizes[source])}) => '
                f'{target.split(".")[0]} (free={bytes_to_string(sizes[target])})"'
            )
        )
        if ARGS.doit:
            try:
                API.compute.live_migrate_server(id, host=target)
                if not ARGS.nowait:
                    API.compute.wait_for_server(server._openstack_object)
                    LOG.info("Move successful")
            except (
                openstack.exceptions.ConflictException,
                openstack.exceptions.BadRequestException,
            ) as e:
                LOG.error(e)
                LOG.error("Move failed")

    with open("moves.sh", "w") as f:

        print("export OS_COMPUTE_API_VERSION=2.30", file=f)

        print(f"# moves: {len(moves)}", file=f)
        for i, (server, target) in enumerate(moves):
            id, name, source, memory = (
                server.id,
                server.name,
                server.host,
                server.memory_size,
            )
            print("date +%H:%M:%S", file=f)
            print(
                (
                    f'echo "{i+1}/{len(moves)} {name} (vm={bytes_to_string(memory)}) '
                    f'{source.split(".")[0]} ({bytes_to_string(sizes[source])}) => '
                    f'{target.split(".")[0]} ({bytes_to_string(sizes[target])})"'
                ),
                file=f,
            )
            wait = "" if ARGS.nowait else "--wait"
            print(
                f"openstack server migrate {wait} --live-migration --host {target} {id}",
                file=f,
            )
            sizes[source] += memory
            sizes[target] -= memory

    return ga_hosts


if ARGS.placement:
    placement()
else:
    display(HYPERVISORS)

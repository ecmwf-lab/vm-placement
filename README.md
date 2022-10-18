# vm-placement

A script that uses [genetic algorithms](https://pygad.readthedocs.io/en/latest/) to determine the best placement of VMs on a series of hypervisors based on affinity rules. **Disclaimer:** this script is experimental and provided *as-is*.

## Installation

Simply run:

    $ pip install -r requirements.txt

## Usage

To use the script, you first need to login to you OpenStack instance. You will need the rigths to list all instances from all projects:

    $ source admin-openrc.sh

See the [OpenStack documentation](https://docs.openstack.org/newton/user-guide/common/cli-set-environment-variables-using-openstack-rc.html) on how to create an `openrc.sh` file.

Then run the script:

    $ ./vm-placement.py

This will produce a map of all VMs and their hypervisors. To also get the I/O diagnostics, run:

    $ ./vm-placement.py --diagnostics


Before you can use the script to place the VMs on the hypervisors in an optimal fashion, you need to edit the file `vm-placement.yaml` to discribe affinities. There are two sorts of affinities:

- VMs that must be placed on the same hypervisor;
- VMs that must be placed on different hypervisors.

The setup below is used to organise the VMs in the [Climate Data Store](https://cds.climate.copernicus.eu).


```yaml
---
affinity:
  mars.*:
    never_with: mars.*

  .*prod.*:
    never_with: mars.*

  cdsprod-compute-.*:
    never_with: cdsprod-compute-.*

  (\w+)-compute-(\w+)-.*:
    always_with: (\w+)-download-(\w+)-.*

  (\w+)-download-(\w+)-.*:
    always_with: (\w+)-compute-(\w+)-.*

  (\w+)-brokerdb-.*:
    never_with: (\w+)-brokerdb-.*
```
- VMs whose names start with `mars` should never be placed on the hypervisors with other VMs also starting with `mars`.
- VMs that contain `prod` in their names must not be placed on the same hypervisors as the ones whose name starts with `mars`.
- VMs with `-compute-` in their names should be placed on the same hypervisors as the ones with `-download-` in their names, assuming the parts captured by the parentheses are the same. `cdsprod-compute-0001-af98` will match `cdsprod-download-0001-34ba`.


Then run:

    $ ./vm-placement.py --placement

This will run a genetic algorithm to find the best placement based on the list of affinity provided. In addition, the algorithm will try to spread out the VMs on all the hypervisors so that the CPU usage is uniform across all the system.

The scripts supports several options, the most important ones are `--generations` and `--population` that controls the number of generation and the size of the population used by the genetic algorithm. The larger these numbers, the more likely the best solution is found.

The script will create a file called `moves.sh` in the current directory. This file contains a series of calls to the openstack command line tool that will move the VMs to their optimal location.

    $ bash ./moves.sh

### Options

For more options, use:

    $ ./vm-placement.py --help



### License
[Apache License 2.0](LICENSE) In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.

### References

```
Ahmed Fawzy Gad. PyGAD: An Intuitive Genetic Algorithm Python Library (2021).
https://doi.org/10.48550/arXiv.2106.06158
```

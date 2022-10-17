# vm-placement

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



<!-- ![OpenStack menu](menu.png) -->

### Disclaimer

This script is experimental and provided *as-is*.

### License
[Apache License 2.0](LICENSE) In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.

import subprocess
import json
from toposort import toposort_flatten

def get_dependency_graph():
  output = subprocess.check_output(['pipdeptree', '--json-tree'])
  deps_list = json.loads(output)
  graph = {}
  versions = {}

  def process_node(node):
    name = node['package_name']
    versions[name] = node['installed_version']
    graph.setdefault(name, set())
    for dep in node['dependencies']:
      graph[name].add(dep['package_name'])
      # Recursively process dependencies
      process_node(dep)

  for node in deps_list:
    process_node(node)

  return graph, versions

def get_toposort_order(graph):
  return toposort_flatten(graph)

def main():
  graph, versions = get_dependency_graph()
  sorted_pkgs = get_toposort_order(graph)
  # Write the sorted package list to 'requirements-core.txt'
  with open("requirements-core.txt", "w") as outfile:
    for pkg in sorted_pkgs:
      outfile.write(f"{pkg}=={versions[pkg]}\n")


if __name__ == "__main__":
  main()
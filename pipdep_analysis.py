import subprocess
import json
from toposort import toposort_flatten

# Define packages that must be installed first
PRIORITY_PACKAGES = {
    'torch': '2.5.1',
}

def get_dependency_graph():
  output = subprocess.check_output(['pipdeptree', '--json-tree'])
  deps_list = json.loads(output)
  graph = {}
  versions = {}

  def process_node(node):
    name = node['package_name']
    versions[name] = node['installed_version']
    graph.setdefault(name, set())
    for dep in node.get('dependencies', []):
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
    # First write priority packages
    for pkg, version in PRIORITY_PACKAGES.items():
      if pkg in versions:
        outfile.write(f"{pkg}=={versions[pkg]}\n")
        sorted_pkgs.remove(pkg)  # Remove from main list to avoid duplicates
    
    # Then write the rest in dependency order
    for pkg in sorted_pkgs:
      if pkg not in PRIORITY_PACKAGES and pkg != 'atpgllm' and pkg != 'pygame':  # Skip if already written
        outfile.write(f"{pkg}=={versions[pkg]}\n")

if __name__ == "__main__":
  main()
# 导入 jsonlite 包
library(jsonlite)
library(networkD3)
library(webshot)
library(htmlwidgets)
nodes <- read.csv('analysis/case2/results//nodes.csv', row.names = 1)
links <- read.csv('analysis/case2/results/links.csv', row.names = 1)
nodes_color_json <- fromJSON("/home/hht/Myapps/unimap/plot_fig/fig2/node_color.json")
nodes_color <- 'd3.scaleOrdinal(domain = names(nodes_color_json), range = unlist(nodes_color_json))'

my_color <- 'd3.scaleOrdinal() .domain(nodes_color_json$celltype) .range(nodes_color_json$color)'

# ----------color----------

# ----------Plot----------
sankey <- sankeyNetwork(Links = links, Nodes = nodes, Source = "source",
             Target = "target", Value = "value", NodeID = "name",
             units = "TWh", fontSize = 12, nodeWidth = 30)

saveWidget(sankey, "results/sankey.html")

webshot("analysis/case2/results/sankey.html", "analysis/case2/results/sankey.pdf")
webshot("analysis/case2/results/sankey.html", "analysis/case2/results/sankey.png", vwidth = 800, vheight = 600, delay = 0.2, zoom = 2)
getwd()

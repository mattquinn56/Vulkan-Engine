#include "stb_image.h"
#include <fstream>
#include <vk_loader.h>

#include "vk_engine.h"
#include "vk_raytracer.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/util.hpp>
#include <nlohmann/json.hpp>
std::optional<AllocatedImage> load_image(VulkanEngine* engine, fastgltf::Asset& asset, fastgltf::Image& image)
{
    AllocatedImage newImage{};

    int width, height, nrChannels;

    std::visit(fastgltf::visitor{
                   [](auto& arg) {},
                   [&](fastgltf::sources::URI& filePath) {
                       assert(filePath.fileByteOffset == 0); // stbi cannot start at an offset
                       assert(filePath.uri.isLocalPath());   // only local files are supported

                       const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
                       unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                       if (data) {
                           VkExtent3D imagesize;
                           imagesize.width = width;
                           imagesize.height = height;
                           imagesize.depth = 1;

                           newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM,
                                                           VK_IMAGE_USAGE_SAMPLED_BIT, false);

                           stbi_image_free(data);
                       }
                   },
                   [&](fastgltf::sources::Vector& vector) {
                       unsigned char* data = stbi_load_from_memory(
                           vector.bytes.data(), static_cast<int>(vector.bytes.size()), &width, &height, &nrChannels, 4);
                       if (data) {
                           VkExtent3D imagesize;
                           imagesize.width = width;
                           imagesize.height = height;
                           imagesize.depth = 1;

                           newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM,
                                                           VK_IMAGE_USAGE_SAMPLED_BIT, false);

                           stbi_image_free(data);
                       }
                   },
                   [&](fastgltf::sources::BufferView& view) {
                       auto& bufferView = asset.bufferViews[view.bufferViewIndex];
                       auto& buffer = asset.buffers[bufferView.bufferIndex];

                       std::visit(fastgltf::visitor{
                                      // LoadExternalBuffers means every buffer is already
                                      // resolved to a vector, so only that case matters.
                                      [](auto& arg) {},
                                      [&](fastgltf::sources::Vector& vector) {
                                          unsigned char* data = stbi_load_from_memory(
                                              vector.bytes.data() + bufferView.byteOffset,
                                              static_cast<int>(bufferView.byteLength), &width, &height, &nrChannels, 4);
                                          if (data) {
                                              VkExtent3D imagesize;
                                              imagesize.width = width;
                                              imagesize.height = height;
                                              imagesize.depth = 1;

                                              newImage = engine->create_image(data, imagesize, VK_FORMAT_R8G8B8A8_UNORM,
                                                                              VK_IMAGE_USAGE_SAMPLED_BIT, false);

                                              stbi_image_free(data);
                                          }
                                      }},
                                  buffer.data);
                   },
               },
               image.data);

    // A null image indicates that every supported loading path failed.
    if (newImage.image == VK_NULL_HANDLE) {
        return {};
    } else {
        return newImage;
    }
}
VkFilter extract_filter(fastgltf::Filter filter)
{
    switch (filter) {
    // nearest samplers
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
        return VK_FILTER_NEAREST;

    // linear samplers
    case fastgltf::Filter::Linear:
    case fastgltf::Filter::LinearMipMapNearest:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_FILTER_LINEAR;
    }
}

VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter)
{
    switch (filter) {
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::LinearMipMapNearest:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;

    case fastgltf::Filter::NearestMipMapLinear:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

std::vector<RenderLight> load_lights(std::string filePath)
{
    std::ifstream inFile(filePath);
    std::stringstream strStream;

    if (!inFile.is_open()) {
        fmt::print(stderr, "Failed to open file: {}\n", filePath);
        return {};
    }

    strStream << inFile.rdbuf();
    std::string jsonData = strStream.str();

    auto j = nlohmann::json::parse(jsonData);
    std::vector<RenderLight> lights = {};

    for (const auto& item : j["lights"]) {
        RenderLight light = {};

        // Type tag packs into color.a; see RenderLight in vk_types.h.
        if (item["type"] == "point") {
            light.color.a = 0.0f;
        } else if (item["type"] == "ambient") {
            light.color.a = 1.0f;
        } else if (item["type"] == "directional") {
            light.color.a = 2.0f;
        } else if (item["type"] == "area") {
            light.color.a = 3.0f;
        } else {
            fmt::print(stderr, "Unknown light type: {}\n", item["type"].dump());
            light.color.a = -1.0f;
        }

        light.position.a = item["intensity"];

        light.color.r = item["color"][0] / 255.0f;
        light.color.g = item["color"][1] / 255.0f;
        light.color.b = item["color"][2] / 255.0f;

        if (item["type"] == "point") {
            light.position.x = item["position"][0];
            light.position.y = item["position"][1];
            light.position.z = item["position"][2];
        } else if (item["type"] == "directional") {
            light.position.x = item["direction"][0];
            light.position.y = item["direction"][1];
            light.position.z = item["direction"][2];
        }

        if (item["type"] == "area") {
            light.v0.x = item["vertices"][0][0];
            light.v0.y = item["vertices"][0][1];
            light.v0.z = item["vertices"][0][2];

            light.v1.x = item["vertices"][1][0];
            light.v1.y = item["vertices"][1][1];
            light.v1.z = item["vertices"][1][2];

            light.position.x = item["vertices"][2][0];
            light.position.y = item["vertices"][2][1];
            light.position.z = item["vertices"][2][2];
        }

        lights.push_back(light);
    }

    return lights;
}

std::optional<std::shared_ptr<LoadedGLTF>> load_gltf(VulkanEngine* engine, std::string_view filePath)
{
    fmt::println("Loading GLTF: {}", filePath);

    std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
    scene->creator = engine;
    LoadedGLTF& file = *scene.get();

    fastgltf::Parser parser{};

    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble |
                                 fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    fastgltf::Asset gltf;

    std::filesystem::path path = filePath;

    auto type = fastgltf::determineGltfFileType(&data);
    if (type == fastgltf::GltfType::glTF) {
        auto load = parser.loadGLTF(&data, path.parent_path(), gltfOptions);
        if (load) {
            gltf = std::move(load.get());
        } else {
            fmt::print(stderr, "Failed to load glTF: {}\n", fastgltf::to_underlying(load.error()));
            return {};
        }
    } else if (type == fastgltf::GltfType::GLB) {
        auto load = parser.loadBinaryGLTF(&data, path.parent_path(), gltfOptions);
        if (load) {
            gltf = std::move(load.get());
        } else {
            fmt::print(stderr, "Failed to load glTF: {}\n", fastgltf::to_underlying(load.error()));
            return {};
        }
    } else {
        fmt::print(stderr, "Failed to determine glTF container\n");
        return {};
    }
    // The asset contents provide an accurate descriptor count estimate.
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
                                                                     {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
                                                                     {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};

    file.descriptorPool.init_pools(engine->_device, static_cast<uint32_t>(gltf.materials.size()), sizes);

    // load samplers
    for (fastgltf::Sampler& sampler : gltf.samplers) {

        VkSamplerCreateInfo sampl = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .pNext = nullptr};
        sampl.maxLod = VK_LOD_CLAMP_NONE;
        sampl.minLod = 0;

        sampl.magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
        sampl.minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        sampl.mipmapMode = extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        VkSampler newSampler;
        vkCreateSampler(engine->_device, &sampl, nullptr, &newSampler);

        file.samplers.push_back(newSampler);
    }
    // temporal arrays for all the objects to use while creating the GLTF data
    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<AllocatedImage> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;

    // load all textures
    for (fastgltf::Image& image : gltf.images) {
        std::optional<AllocatedImage> img = load_image(engine, gltf, image);

        if (img.has_value()) {
            images.push_back(*img);
            file.images[image.name.c_str()] = *img;
        } else {
            // Fall back to a white texture so one bad image does not fail the load.
            images.push_back(engine->_errorCheckerboardImage);
            fmt::print(stderr, "glTF failed to load texture {}\n", image.name);
        }
    }
    // create buffer to hold the material data
    file.materialDataBuffer =
        engine->create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants) * gltf.materials.size(),
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    int data_index = 0;
    GLTFMetallic_Roughness::MaterialConstants* sceneMaterialConstants =
        (GLTFMetallic_Roughness::MaterialConstants*)file.materialDataBuffer.info.pMappedData;
    engine->_rayTracer->_colorTextures.clear();
    engine->_rayTracer->_metalRoughTextures.clear();
    engine->_rayTracer->_colorSamplers.clear();
    engine->_rayTracer->_metalRoughSamplers.clear();
    for (fastgltf::Material& mat : gltf.materials) {
        std::shared_ptr<GLTFMaterial> newMat = std::make_shared<GLTFMaterial>();
        materials.push_back(newMat);
        file.materials[mat.name.c_str()] = newMat;

        GLTFMetallic_Roughness::MaterialConstants constants;
        constants.colorFactors.x = mat.pbrData.baseColorFactor[0];
        constants.colorFactors.y = mat.pbrData.baseColorFactor[1];
        constants.colorFactors.z = mat.pbrData.baseColorFactor[2];
        constants.colorFactors.w = mat.pbrData.baseColorFactor[3];

        constants.metalRoughFactors.x = mat.pbrData.metallicFactor;
        constants.metalRoughFactors.y = mat.pbrData.roughnessFactor;
        // write material parameters to buffer
        sceneMaterialConstants[data_index] = constants;

        MaterialPass passType = MaterialPass::MainColor;
        if (mat.alphaMode == fastgltf::AlphaMode::Blend) {
            passType = MaterialPass::Transparent;
        }

        GLTFMetallic_Roughness::MaterialResources materialResources;
        // default the material textures
        materialResources.colorImage = engine->_whiteImage;
        materialResources.colorSampler = engine->_defaultSamplerLinear;
        materialResources.metalRoughImage = engine->_whiteImage;
        materialResources.metalRoughSampler = engine->_defaultSamplerLinear;

        // set the uniform buffer for the material data
        materialResources.dataBuffer = file.materialDataBuffer.buffer;
        materialResources.dataBufferOffset = data_index * sizeof(GLTFMetallic_Roughness::MaterialConstants);
        // grab textures from gltf file
        if (mat.pbrData.baseColorTexture.has_value()) {
            size_t img = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
            size_t sampler = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();

            materialResources.colorImage = images[img];
            materialResources.colorSampler = file.samplers[sampler];
        }
        // build material
        newMat->data = engine->_metalRoughMaterial.write_material(engine->_device, passType, materialResources,
                                                                  file.descriptorPool);

        // upload RT material
        VulkanRayTracer::MaterialRT rtMat;
        rtMat.colorFactors = constants.colorFactors;
        rtMat.metalRoughFactors = constants.metalRoughFactors;
        rtMat.textureID = data_index;
        newMat->materialAddressRT = engine->_rayTracer->upload_material(rtMat);

        // accumulate images for ray tracing textures
        engine->_rayTracer->_colorTextures.push_back(materialResources.colorImage.imageView);
        engine->_rayTracer->_metalRoughTextures.push_back(materialResources.metalRoughImage.imageView);
        engine->_rayTracer->_colorSamplers.push_back(materialResources.colorSampler);
        engine->_rayTracer->_metalRoughSamplers.push_back(materialResources.metalRoughSampler);

        data_index++;
    }

    // Reuse these vectors across meshes to reduce allocations.
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh& mesh : gltf.meshes) {
        std::shared_ptr<MeshAsset> newmesh = std::make_shared<MeshAsset>();
        meshes.push_back(newmesh);
        file.meshes[mesh.name.c_str()] = newmesh;
        newmesh->name = mesh.name;

        // Reset temporary geometry so adjacent meshes are not merged.
        indices.clear();
        vertices.clear();

        for (auto&& p : mesh.primitives) {
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

            size_t initial_vtx = vertices.size();

            // load indexes
            {
                fastgltf::Accessor& indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexaccessor, [&](std::uint32_t idx) {
                    indices.push_back(idx + static_cast<uint32_t>(initial_vtx));
                });
            }

            // load vertex positions
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor, [&](glm::vec3 v, size_t index) {
                    Vertex newvtx;
                    newvtx.position = v;
                    newvtx.normal = {1, 0, 0};
                    newvtx.color = glm::vec4{1.f};
                    newvtx.uvX = 0;
                    newvtx.uvY = 0;
                    vertices[initial_vtx + index] = newvtx;
                });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                    gltf, gltf.accessors[(*normals).second],
                    [&](glm::vec3 v, size_t index) { vertices[initial_vtx + index].normal = v; });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
                                                              [&](glm::vec2 v, size_t index) {
                                                                  vertices[initial_vtx + index].uvX = v.x;
                                                                  vertices[initial_vtx + index].uvY = v.y;
                                                              });
            }

            // load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec4>(
                    gltf, gltf.accessors[(*colors).second],
                    [&](glm::vec4 v, size_t index) { vertices[initial_vtx + index].color = v; });
            }

            if (p.materialIndex.has_value()) {
                newSurface.material = materials[p.materialIndex.value()];
            } else {
                newSurface.material = materials[0];
            }

            glm::vec3 minpos = vertices[initial_vtx].position;
            glm::vec3 maxpos = vertices[initial_vtx].position;
            for (size_t i = initial_vtx; i < vertices.size(); i++) {
                minpos = glm::min(minpos, vertices[i].position);
                maxpos = glm::max(maxpos, vertices[i].position);
            }

            newSurface.bounds.origin = (maxpos + minpos) / 2.f;
            newSurface.bounds.extents = (maxpos - minpos) / 2.f;
            newSurface.bounds.sphereRadius = glm::length(newSurface.bounds.extents);
            newmesh->surfaces.push_back(newSurface);
        }

        newmesh->meshBuffers = engine->upload_mesh(indices, vertices);
    }
    // load all nodes and their meshes
    for (fastgltf::Node& node : gltf.nodes) {
        std::shared_ptr<Node> newNode;

        // Nodes carrying geometry become MeshNodes; the rest are transform-only.
        if (node.meshIndex.has_value()) {
            newNode = std::make_shared<MeshNode>();
            static_cast<MeshNode*>(newNode.get())->mesh = meshes[*node.meshIndex];
        } else {
            newNode = std::make_shared<Node>();
        }
        newNode->engine = engine;

        // glTF node names are not required to be unique; disambiguate with a suffix.
        std::pmr::string nodeName = node.name;
        while (file.nodeNames.find(newNode) != file.nodeNames.end()) {
            nodeName = fmt::format("{}_{}", node.name.c_str(), file.nodeNames.size());
        }

        file.nodeNames[newNode] = nodeName;
        nodes.push_back(newNode);

        std::visit(fastgltf::visitor{[&](fastgltf::Node::TransformMatrix matrix) {
                                         memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
                                     },
                                     [&](fastgltf::Node::TRS transform) {
                                         glm::vec3 tl(transform.translation[0], transform.translation[1],
                                                      transform.translation[2]);
                                         glm::quat rot(transform.rotation[3], transform.rotation[0],
                                                       transform.rotation[1], transform.rotation[2]);
                                         glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                                         glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
                                         glm::mat4 rm = glm::toMat4(rot);
                                         glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

                                         newNode->localTransform = tm * rm * sm;
                                     }},
                   node.transform);
    }
    // run loop again to setup transform hierarchy
    for (int i = 0; i < gltf.nodes.size(); i++) {
        fastgltf::Node& node = gltf.nodes[i];
        std::shared_ptr<Node>& sceneNode = nodes[i];

        for (auto& c : node.children) {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    // find the top nodes, with no parents
    for (auto& node : nodes) {
        if (node->parent.lock() == nullptr) {
            file.topNodes.push_back(node);
            node->refresh_transform(glm::mat4{1.f});
        }
    }
    return scene;
}

void LoadedGLTF::draw(const glm::mat4& topMatrix, DrawContext& ctx)
{
    // Create renderables from the scene nodes.
    for (auto& n : topNodes) {
        n->draw(topMatrix, ctx);
    }
}

void LoadedGLTF::destroy_owned_resources()
{
    if (creator == nullptr) {
        return;
    }

    VkDevice dv = creator->_device;

    for (auto& [k, v] : meshes) {

        creator->destroy_buffer(v->meshBuffers.indexBuffer);
        creator->destroy_buffer(v->meshBuffers.vertexBuffer);
    }

    for (auto& [k, v] : images) {

        if (v.image == creator->_errorCheckerboardImage.image) {
            // Default images are owned by the engine.
            continue;
        }
        creator->destroy_image(v);
    }

    for (auto& sampler : samplers) {
        vkDestroySampler(dv, sampler, nullptr);
    }

    descriptorPool.destroy_pools(dv);

    creator->destroy_buffer(materialDataBuffer);
}

// Minimal ELF32 definitions for hosts without <elf.h> (macOS). Values per the System V ABI.
#pragma once
#if __has_include(<elf.h>)
#include <elf.h>
#else
#include <cstdint>
#define EI_NIDENT 16
#define EI_MAG0 0
#define EI_MAG1 1
#define EI_MAG2 2
#define EI_MAG3 3
#define EI_CLASS 4
#define EI_DATA 5
#define ELFMAG0 0x7f
#define ELFMAG1 'E'
#define ELFMAG2 'L'
#define ELFMAG3 'F'
#define ELFCLASS32 1
#define ELFDATA2LSB 1
#define PT_LOAD 1
#define SHT_NOBITS 8
#define SHF_ALLOC 0x2
#define SHF_EXECINSTR 0x4
struct Elf32_Ehdr {
    unsigned char e_ident[EI_NIDENT];
    uint16_t e_type, e_machine;
    uint32_t e_version, e_entry, e_phoff, e_shoff, e_flags;
    uint16_t e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx;
};
struct Elf32_Phdr { uint32_t p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, p_align; };
struct Elf32_Shdr {
    uint32_t sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize;
};
#endif
